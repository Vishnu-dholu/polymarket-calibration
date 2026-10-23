"""Step 5: longshot/favourite strategy backtest with cost sensitivity, event bootstrap, time split."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analysis import RESULTS, cluster_weights, event_sums, load

COSTS = [0.0, 0.01, 0.02, 0.03]          # extra cost per share, in dollars (covers spread/slippage/fees)

# name -> (side, filter). Rules come from the project brief, chosen before looking at P&L.
RULES = {
    "Buy NO when YES < 10c":       ("NO",  lambda d: d.price < 0.10),
    "Buy NO when YES < 5c":        ("NO",  lambda d: d.price < 0.05),
    "Buy NO when 5c <= YES < 10c": ("NO",  lambda d: (d.price >= 0.05) & (d.price < 0.10)),
    "Buy YES when YES > 90c":      ("YES", lambda d: d.price > 0.90),
    "Buy YES when YES > 95c":      ("YES", lambda d: d.price > 0.95),
}
MAIN_RULES = ["Buy NO when YES < 10c", "Buy YES when YES > 90c"]


def trade_pnl(d, side, cost):
    """P&L per share held to resolution. Payout is $1 if the bought side wins, else $0."""
    if side == "NO":
        entry = 1 - d.price + cost
        pnl = (1 - d.outcome) - entry
    else:
        entry = d.price + cost
        pnl = d.outcome - entry
    return pnl.to_numpy(), entry.to_numpy()


def summarize(d, side, cost):
    """Mean P&L per share in cents, with a 95% CI from resampling whole events."""
    pnl, entry = trade_pnl(d, side, cost)
    codes, E, W = cluster_weights(d.cluster)
    S = event_sums(codes, E, np.column_stack([pnl, np.ones(len(d))]))
    boot = (W @ S[:, 0]) / (W @ S[:, 1])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": len(d), "events": E, "mean_pnl_c": pnl.mean() * 100,
            "lo_c": lo * 100, "hi_c": hi * 100, "total_$": pnl.sum(),
            "return_on_capital_pct": pnl.sum() / entry.sum() * 100,
            "win_rate": (pnl > 0).mean()}


def plot_cost_sensitivity(res, h, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for rule in MAIN_RULES:
        t = res[(res.horizon == h) & (res.rule == rule) & (res.subset == "all")]
        ax.errorbar(t.cost_c, t.mean_pnl_c, yerr=[t.mean_pnl_c - t.lo_c, t.hi_c - t.mean_pnl_c],
                    marker="o", capsize=3, label=rule)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Extra cost per share (cents)")
    ax.set_ylabel("Mean P&L per share (cents)")
    ax.set_title(f"Cost sensitivity, {h} snapshot (95% CI, bootstrap by event)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equity(d, h, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for rule in MAIN_RULES:
        side, fn = RULES[rule]
        sub = d[fn(d)].sort_values("ref_time")
        for cost in (0.0, 0.02):
            pnl, _ = trade_pnl(sub, side, cost)
            ax.plot(sub.ref_time, np.cumsum(pnl), label=f"{rule}, +{cost * 100:.0f}c")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Scheduled end date")
    ax.set_ylabel("Cumulative P&L ($, 1 share per market)")
    ax.set_title(f"Equity curve, {h} snapshot")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    rows = []
    data = {}
    for h in ("24h", "7d"):
        d = load(h)
        d["half"] = np.where(d.ref_time <= d.ref_time.median(), "older", "newer")
        data[h] = d
        print(f"{h}: {len(d)} markets | older half ends {d.ref_time.median().date()} | "
              f"fees enabled: {d.fees_enabled.fillna(False).astype(bool).mean():.1%}")
        for rule, (side, fn) in RULES.items():
            sub = d[fn(d)]
            for cost in COSTS:
                for label, g in [("all", sub), ("older", sub[sub.half == "older"]),
                                 ("newer", sub[sub.half == "newer"])]:
                    if len(g) == 0:
                        continue
                    r = summarize(g, side, cost)
                    r.update(horizon=h, rule=rule, cost_c=cost * 100, subset=label)
                    rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "backtest_results.csv", index=False)
    cols = ["subset", "cost_c", "n", "events", "mean_pnl_c", "lo_c", "hi_c",
            "return_on_capital_pct", "win_rate"]
    for h in ("24h", "7d"):
        for rule in RULES:
            t = res[(res.horizon == h) & (res.rule == rule)]
            print(f"\n=== {h} | {rule} ===")
            print(t[t.subset == "all"][cols].round(3).to_string(index=False))
            print("time split (cost 0c and 2c):")
            print(t[(t.subset != "all") & (t.cost_c.isin([0, 2]))].sort_values(["cost_c", "subset"])
                  [cols].round(3).to_string(index=False))
        plot_cost_sensitivity(res, h, RESULTS / f"cost_sensitivity_{h}.png")
        plot_equity(data[h], h, RESULTS / f"equity_{h}.png")
    print("\nSaved results/backtest_results.csv and PNGs")