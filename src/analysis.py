"""Calibration analysis: event-clustered bootstrap, calibration tables, Brier score, plots."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from categories import assign_category

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"          # tracked in git (small CSVs and PNGs)

BINS = [0, .05, .10, .20, .40, .60, .80, .90, .95, 1.0]
LONGSHOT_BINS = [0, .01, .02, .05, .10]
N_BOOT = 2000
SEED = 42


# ---------- data ----------
def load(horizon="24h"):
    """One row per market with a usable point-in-time snapshot at this horizon."""
    snaps = pd.read_parquet(DATA / "snapshots.parquet").drop(columns=["category"], errors="ignore")
    ev = pd.read_parquet(DATA / "events.parquet")[["event_id", "tag_slugs"]]
    d = snaps.merge(ev, on="event_id", how="left")
    d["category"] = [assign_category(t, q) for t, q in zip(d.tag_slugs, d.question)]
    d["cluster"] = d.event_id.fillna(d.id).astype(str)   # bootstrap unit = event
    d["year"] = d.ref_time.dt.year
    d = d[d[f"status_{horizon}"] == "ok"].copy()
    d["price"] = d[f"price_{horizon}"]
    assert d.price.between(0, 1).all(), "price outside [0, 1]"
    return d.reset_index(drop=True)


# ---------- event-clustered bootstrap ----------
def cluster_weights(clusters, n_boot=N_BOOT, seed=SEED):
    """Resample whole events with replacement. Returns (event code per row, #events, weights B x E)."""
    codes, uniq = pd.factorize(clusters)
    E = len(uniq)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(E, np.full(E, 1 / E), size=n_boot)
    return codes, E, W


def event_sums(codes, E, values):
    """Sum row-level values (n x K) into event-level totals (E x K)."""
    out = np.zeros((E, values.shape[1]))
    np.add.at(out, codes, values)
    return out


# ---------- calibration ----------
def calibration_table(d, bins=BINS):
    """Per price bucket: n, events, avg price, actual YES rate, gap, and 95% event-bootstrap CIs."""
    codes, E, W = cluster_weights(d.cluster)
    bucket = pd.cut(d.price, bins, include_lowest=True)
    labels = bucket.cat.categories
    K = len(labels)
    onehot = np.zeros((len(d), K))
    onehot[np.arange(len(d)), bucket.cat.codes.to_numpy()] = 1

    N = event_sums(codes, E, onehot)
    Y = event_sums(codes, E, onehot * d.outcome.to_numpy()[:, None])
    P = event_sums(codes, E, onehot * d.price.to_numpy()[:, None])
    n, y, p = N.sum(0), Y.sum(0), P.sum(0)

    bn, by, bp = W @ N, W @ Y, W @ P
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate_b, gap_b = by / bn, by / bn - bp / bn
        rate_lo, rate_hi = np.nanpercentile(rate_b, [2.5, 97.5], axis=0)
        gap_lo, gap_hi = np.nanpercentile(gap_b, [2.5, 97.5], axis=0)
        avg_price, yes_rate = p / n, y / n

    return pd.DataFrame({
        "bucket": labels.astype(str), "n": n.astype(int), "events": (N > 0).sum(0),
        "avg_price": avg_price, "yes_rate": yes_rate, "gap": yes_rate - avg_price,
        "rate_lo": rate_lo, "rate_hi": rate_hi, "gap_lo": gap_lo, "gap_hi": gap_hi,
    })


# ---------- Brier score ----------
def brier_ci(d):
    """Brier score with an event-clustered 95% CI. Returns (brier, lo, hi)."""
    codes, E, W = cluster_weights(d.cluster)
    se = ((d.price - d.outcome) ** 2).to_numpy()
    S = event_sums(codes, E, np.column_stack([se, np.ones(len(d))]))
    boot = W @ S
    lo, hi = np.percentile(boot[:, 0] / boot[:, 1], [2.5, 97.5])
    return se.mean(), lo, hi


def brier_table(d, by=None):
    groups = [("ALL", d)] if by is None else list(d.groupby(by))
    rows = []
    for name, g in groups:
        b, lo, hi = brier_ci(g)
        base = g.outcome.mean() * (1 - g.outcome.mean())   # Brier of always predicting the base rate
        rows.append({"group": name, "n": len(g), "events": g.cluster.nunique(),
                     "brier": b, "lo": lo, "hi": hi, "base_rate_brier": base,
                     "yes_rate": g.outcome.mean()})
    return pd.DataFrame(rows)


# ---------- plot ----------
def plot_calibration(tab, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tab[tab.n > 0]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    yerr = [t.yes_rate - t.rate_lo, t.rate_hi - t.yes_rate]
    ax.errorbar(t.avg_price, t.yes_rate, yerr=yerr, fmt="o", capsize=3,
                label="price buckets (95% CI, bootstrap by event)")
    for _, r in t.iterrows():
        ax.annotate(f"n={int(r.n)}", (r.avg_price, r.yes_rate),
                    textcoords="offset points", xytext=(6, -11), fontsize=7)
    ax.set_xlabel("Average market price (implied probability)")
    ax.set_ylabel("Actual YES rate")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- run ----------
if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    for h in ("24h", "7d"):
        d = load(h)
        print(f"\n{'=' * 20} horizon {h}: {len(d)} markets, {d.cluster.nunique()} events {'=' * 20}")

        tab = calibration_table(d)
        print("\nCalibration (all prices):")
        print(tab.round(3).to_string(index=False))
        tab.to_csv(RESULTS / f"calibration_{h}.csv", index=False)
        plot_calibration(tab, f"Polymarket calibration, {h} before scheduled end",
                         RESULTS / f"calibration_{h}.png")

        ls = calibration_table(d[d.price <= 0.10], LONGSHOT_BINS)
        print("\nLongshot detail (price <= 10c):")
        print(ls.round(3).to_string(index=False))
        ls.to_csv(RESULTS / f"longshot_{h}.csv", index=False)

        print("\nBrier overall:")
        print(brier_table(d).round(4).to_string(index=False))
        by_cat = brier_table(d, "category")
        print("\nBrier by category:")
        print(by_cat.round(4).to_string(index=False))
        by_cat.to_csv(RESULTS / f"brier_by_category_{h}.csv", index=False)
        print("\nBrier by year:")
        print(brier_table(d, "year").round(4).to_string(index=False))

    print("\nSaved CSVs and PNGs to results/")