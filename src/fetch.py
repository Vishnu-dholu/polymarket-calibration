"""Fetch resolved Polymarket binary markets and price snapshots, with on-disk caching."""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com/prices-history"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"
SLEEP = 0.2  # seconds between uncached requests

HORIZONS = {"24h": pd.Timedelta(hours=24), "7d": pd.Timedelta(days=7)}
LOOKBACKS = (pd.Timedelta(days=2), pd.Timedelta(days=6))  # CLOB rejects long windows


# ---------- networking ----------
def _make_session():
    s = requests.Session()
    retry = Retry(total=6, connect=6, read=3, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "polymarket-calibration-study/0.1",
                      "Accept": "application/json"})
    return s


_S = _make_session()


def _get(url, params):
    for attempt in range(5):
        try:
            return _S.get(url, params=params, timeout=30)
        except requests.exceptions.ConnectionError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Connection kept failing for {url}")


def cached_json(kind, url, params):
    """GET with a disk cache. Only 200 responses are cached (empty history included)."""
    key = hashlib.md5(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
    path = CACHE / kind / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    r = _get(url, params)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {url} {params} -> {r.text[:200]}")
    data = r.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(SLEEP)
    return data


# ---------- markets ----------
def _ts(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def parse_market(m):
    """Return (row, None) for a clean binary YES/NO market, else (None, drop_reason)."""
    try:
        outcomes = json.loads(m["outcomes"])
        prices = [float(p) for p in json.loads(m["outcomePrices"])]
        tokens = json.loads(m["clobTokenIds"])
    except (KeyError, TypeError, ValueError):
        return None, "unparseable"
    if outcomes != ["Yes", "No"]:
        return None, "not_yes_no"
    if len(tokens) != 2 or len(prices) != 2:
        return None, "bad_tokens_or_prices"
    if prices[0] >= 0.99 and prices[1] <= 0.01:
        outcome = 1
    elif prices[1] >= 0.99 and prices[0] <= 0.01:
        outcome = 0
    else:
        return None, "unresolved_cancelled_or_5050"

    end_time, closed_time, created = _ts(m.get("endDate")), _ts(m.get("closedTime")), _ts(m.get("createdAt"))
    if pd.isna(closed_time) and pd.isna(end_time):
        return None, "no_timestamps"
    ref_time = end_time if not pd.isna(end_time) else closed_time
    events = m.get("events") or [{}]
    return {
        "id": str(m["id"]),
        "question": m.get("question"),
        "category": m.get("category"),
        "event_id": events[0].get("id"),
        "created_time": created,
        "end_time": end_time,
        "closed_time": closed_time,
        "ref_time": ref_time,
        "volume": float(m.get("volumeNum") or m.get("volume") or 0),
        "yes_token": tokens[0],  # outcomes == ["Yes","No"] verified above
        "outcome": outcome,
        "fees_enabled": m.get("feesEnabled"),
        "fee_type": m.get("feeType"),
        "uma": json.dumps(m.get("umaResolutionStatuses")),
    }, None


def fetch_markets(start, end, min_volume, per_window, page=100):
    edges = pd.date_range(start, end, freq="MS")  # use month-start dates
    rows, drops, seen = [], Counter(), set()
    for lo, hi in tqdm(list(zip(edges[:-1], edges[1:])), desc="months"):
        offset = 0
        while offset < per_window:
            limit = min(page, per_window - offset)
            params = {"closed": "true", "limit": limit, "offset": offset,
                      "end_date_min": lo.strftime("%Y-%m-%d"),
                      "end_date_max": hi.strftime("%Y-%m-%d"),
                      "volume_num_min": min_volume,
                      "order": "volumeNum", "ascending": "false"}
            batch = cached_json("gamma", GAMMA, params)
            for m in batch:
                drops["seen"] += 1
                row, reason = parse_market(m)
                if row is None:
                    drops[reason] += 1
                elif row["id"] in seen:
                    drops["duplicate"] += 1
                else:
                    seen.add(row["id"])
                    rows.append(row)
            if len(batch) < limit:
                break
            offset += limit
    df = pd.DataFrame(rows)
    DATA.mkdir(exist_ok=True)
    df.to_parquet(DATA / "markets_raw.parquet", index=False)
    (DATA / "drop_log.json").write_text(json.dumps(drops, indent=2))
    print("\nDrop log:", dict(drops))
    print("Kept:", len(df), "| saved data/markets_raw.parquet")
    return df


# ---------- snapshots ----------
def snapshot(token, created, closed, ref_time, horizon):
    """Point-in-time snapshot. Returns (price, age_hours, status)."""
    target = ref_time - horizon
    if pd.notna(created) and created > target:
        return None, None, "not_yet_created"
    if pd.notna(closed) and closed <= target:
        return None, None, "closed_before_snapshot"
    target_s = int(target.timestamp())
    for lookback in LOOKBACKS:
        params = {"market": token, "startTs": int((target - lookback).timestamp()),
                  "endTs": target_s, "fidelity": 60}
        hist = cached_json("clob", CLOB, params).get("history", [])
        hist = [h for h in hist if h["t"] <= target_s]  # guard against lookahead
        if hist:
            last = hist[-1]
            return float(last["p"]), (target_s - last["t"]) / 3600, "ok"
    return None, None, "no_history"


def fetch_snapshots():
    markets = pd.read_parquet(DATA / "markets_raw.parquet")
    out, errors = [], 0
    for r in tqdm(markets.itertuples(index=False), total=len(markets), desc="snapshots"):
        row = {"id": r.id}
        for name, delta in HORIZONS.items():
            try:
                p, age, status = snapshot(r.yes_token, r.created_time, r.closed_time,
                                          r.ref_time, delta)
            except RuntimeError as e:
                p, age, status, errors = None, None, "error", errors + 1
                tqdm.write(str(e)[:200])
            row[f"price_{name}"], row[f"age_{name}_h"], row[f"status_{name}"] = p, age, status
        out.append(row)
    snaps = markets.merge(pd.DataFrame(out), on="id")
    snaps.to_parquet(DATA / "snapshots.parquet", index=False)
    print(f"\nMarkets: {len(snaps)} | errors: {errors}")
    for name in HORIZONS:
        print(f"\n{name} status counts:\n{snaps[f'status_{name}'].value_counts().to_string()}")
    print("\nSaved data/snapshots.parquet")
# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("markets")
    pm.add_argument("--start", default="2023-01-01")
    pm.add_argument("--end", default="2026-09-01")
    pm.add_argument("--min-volume", type=float, default=5000)
    pm.add_argument("--per-window", type=int, default=150)
    sub.add_parser("snapshots")
    a = ap.parse_args()
    if a.cmd == "markets":
        fetch_markets(a.start, a.end, a.min_volume, a.per_window)
    else:
        fetch_snapshots()