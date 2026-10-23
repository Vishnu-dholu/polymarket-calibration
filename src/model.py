"""Step 6: does a recalibration model beat the raw market price out of sample? Walk-forward by time."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from analysis import RESULTS, cluster_weights, event_sums, load

HORIZONS = {"24h": pd.Timedelta(hours=24), "7d": pd.Timedelta(days=7)}
EPS = 0.001   # probabilities are clipped to [EPS, 1-EPS] for logit and log loss

# name -> (numeric features, categorical features)
FEATURE_SETS = {
    "M1 recalibrate: logit(price)":     (["logit_p"], []),
    "M2 + log(age) + category":         (["logit_p", "log_age"], ["category"]),
    "M3 + log(final volume) [LEAKY]":   (["logit_p", "log_age", "log_vol"], ["category"]),
}


def add_features(d, horizon):
    d = d.copy()
    d["snap_time"] = d.ref_time - HORIZONS[horizon]
    p = d.price.clip(EPS, 1 - EPS)
    d["logit_p"] = np.log(p / (1 - p))
    age_days = (d.snap_time - d.created_time).dt.total_seconds() / 86400   # age AT the snapshot
    d["log_age"] = np.log1p(age_days.clip(lower=0).fillna(age_days.median()))
    d["log_vol"] = np.log(d.volume.clip(lower=1))      # FINAL volume: includes trading after the snapshot
    d["resolved"] = d.closed_time.fillna(d.ref_time)   # when the outcome became known (approx.)
    return d


def make_model(num, cat):
    parts = [("num", StandardScaler(), num)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(C=10, max_iter=2000))


def walk_forward(d, num, cat, min_train=200):
    """Expanding window, quarterly test folds from 2024Q1.
    Train only on markets already resolved before the fold starts; test on markets
    whose snapshot falls inside the fold."""
    edges = pd.date_range("2024-01-01", "2027-01-01", freq="QS", tz="UTC")
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        test = d[(d.snap_time >= lo) & (d.snap_time < hi)]
        train = d[d.resolved <= lo]
        if len(test) == 0 or len(train) < min_train or train.outcome.nunique() < 2:
            continue
        m = make_model(num, cat).fit(train[num + cat], train.outcome)
        parts.append(test.assign(p_model=m.predict_proba(test[num + cat])[:, 1]))
    return pd.concat(parts)


def logloss(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def diff_ci(res, diff):
    """Mean of a per-market difference with a 95% CI from resampling whole events."""
    codes, E, W = cluster_weights(res.cluster)
    S = event_sums(codes, E, np.column_stack([diff, np.ones(len(diff))]))
    boot = (W @ S[:, 0]) / (W @ S[:, 1])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return diff.mean(), lo, hi


def evaluate(res):
    y = res.outcome.to_numpy()
    pm = res.price.clip(EPS, 1 - EPS).to_numpy()
    pr = res.p_model.clip(EPS, 1 - EPS).to_numpy()
    b = diff_ci(res, (pr - y) ** 2 - (pm - y) ** 2)
    l = diff_ci(res, logloss(pr, y) - logloss(pm, y))
    return {"n_test": len(res), "events": res.cluster.nunique(),
            "brier_market": ((pm - y) ** 2).mean(), "brier_model": ((pr - y) ** 2).mean(),
            "d_brier": b[0], "d_brier_lo": b[1], "d_brier_hi": b[2],
            "logloss_market": logloss(pm, y).mean(), "logloss_model": logloss(pr, y).mean(),
            "d_logloss": l[0], "d_ll_lo": l[1], "d_ll_hi": l[2]}


def calibration_slope(d):
    """In-sample, descriptive: logistic fit of outcome on logit(price), SEs clustered by event.
    A perfectly calibrated market has intercept 0 and slope 1."""
    X = sm.add_constant(d[["logit_p"]])
    groups, _ = pd.factorize(d.cluster)
    fit = sm.Logit(d.outcome, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    ci = fit.conf_int()
    for k, target in (("const", 0), ("logit_p", 1)):
        print(f"  {k:8s} {fit.params[k]:+.3f}  95% CI [{ci.loc[k, 0]:+.3f}, {ci.loc[k, 1]:+.3f}]"
              f"   (perfect calibration = {target})")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 250)
    out = []
    for h in HORIZONS:
        d = add_features(load(h), h)
        print(f"\n===== {h}: {len(d)} markets, {d.cluster.nunique()} events =====")
        print("Calibration slope (all data, in-sample):")
        calibration_slope(d)
        for name, (num, cat) in FEATURE_SETS.items():
            res = walk_forward(d, num, cat)
            row = evaluate(res)
            row.update(horizon=h, model=name,
                       test_from=res.snap_time.min().date(), test_to=res.snap_time.max().date())
            out.append(row)
    tab = pd.DataFrame(out)
    cols = ["horizon", "model", "n_test", "events", "brier_market", "brier_model",
            "d_brier", "d_brier_lo", "d_brier_hi", "logloss_market", "logloss_model",
            "d_logloss", "d_ll_lo", "d_ll_hi"]
    print("\nOut-of-sample (walk-forward). d_* = model minus market; NEGATIVE = model better.")
    print(tab[cols].round(4).to_string(index=False))
    print("\ntest window:", tab.test_from.iloc[0], "to", tab.test_to.iloc[0])
    tab.to_csv(RESULTS / "model_comparison.csv", index=False)
    print("Saved results/model_comparison.csv")