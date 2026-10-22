"""Scenario-coverage experiment on UCI Adult (stand-in for a scenario-based benchmark).

Baseline : plain logistic regression; per-scenario aggregate accuracy/macro-F1 is the summary.
Improved : class-balanced, CV-tuned LightGBM with scenario-coverage features
           (scenario size, per-scenario class balance), calibrated-averaged with the baseline.
Main metric: worst-scenario macro-F1 over scenarios with >= 50 test rows (higher is better).
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from lightgbm import LGBMClassifier

SEED = 42
SEEDS = [42, 43, 44]
MIN_ROWS = 50
THRESH = 0.70  # scenarios with worst-case macro-F1 below this are "under-served"
URL = "https://archive.ics.uci.edu/dataset/2/adult"
t0 = time.time()
os.makedirs("figures", exist_ok=True)

# ---- 1. Load data (ucimlrepo, with retry/backoff on HTTP 429/5xx) ----
def load_adult():
    from ucimlrepo import fetch_ucirepo
    for attempt in range(5):
        try:
            return fetch_ucirepo(id=2)
        except Exception as e:  # network / HTTP errors
            wait = 2 ** attempt
            print(f"download failed ({e}); retry in {wait}s")
            time.sleep(wait)
    sys.exit("could not download Adult dataset")

print("loading Adult dataset...")
ds = load_adult()
X = ds.data.features.copy()
y = (ds.data.targets.iloc[:, 0].astype(str).str.strip().str.rstrip(".") == ">50K").astype(int)
print(f"rows: {len(X)}")

# ---- 2. Scenario construction (deterministic rule on categorical attributes) ----
edu = X["education-num"].fillna(0)
band = pd.cut(edu, bins=[-1, 8, 12, 16]/np.array([1,1,1,1]) if False else [-1, 8, 12, 99],
              labels=["low_edu", "mid_edu", "high_edu"]).astype(str)
scenario = band + "_" + X["sex"].fillna("NA").astype(str)

# ---- 3. Single stratified 70/30 split (stratified by income label and scenario) ----
strat = scenario + "_" + y.astype(str)
idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.3, random_state=SEED, stratify=strat)
X_tr, X_te = X.iloc[idx_tr].reset_index(drop=True), X.iloc[idx_te].reset_index(drop=True)
y_tr, y_te = y.iloc[idx_tr].reset_index(drop=True), y.iloc[idx_te].reset_index(drop=True)
s_tr, s_te = scenario.iloc[idx_tr].reset_index(drop=True), scenario.iloc[idx_te].reset_index(drop=True)
print(f"train={len(X_tr)} test={len(X_te)}")

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

def base_pre():
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cat_cols)])

def coverage_feats(Xd, s, size_map, bal_map):
    """Scenario-coverage features: scenario training size and per-scenario positive rate."""
    out = Xd.copy()
    out["scen_size"] = s.map(size_map).fillna(0).values
    out["scen_pos_rate"] = s.map(bal_map).fillna(0.5).values
    return out

def scenario_report(pred, s, y_true):
    """Per-scenario macro-F1 and accuracy; returns worst-scenario macro-F1 over scenarios >= MIN_ROWS."""
    rows = []
    for sc in sorted(s.unique()):
        m = (s == sc).values
        if m.sum() < MIN_ROWS:
            continue
        rows.append({"scenario": sc, "n": int(m.sum()),
                     "macro_f1": f1_score(y_true[m], pred[m], average="macro"),
                     "acc": accuracy_score(y_true[m], pred[m])})
    return pd.DataFrame(rows)

def evaluate(name, pred, proba_s=None):
    rep = scenario_report(pred, s_te, y_te)
    res = {"worst_f1": rep["macro_f1"].min(),
           "overall_f1": f1_score(y_te, pred, average="macro"),
           "n_below": int((rep["macro_f1"] < THRESH).sum())}
    print(f"{name}: worst-scenario macro-F1={res['worst_f1']:.4f} overall macro-F1={res['overall_f1']:.4f} "
          f"scenarios<{THRESH}: {res['n_below']}")
    return res, rep

# ---- 4. BASELINE: logistic regression, per-scenario aggregate reporting ----
# Tune C by 5-fold CV on training data only.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
base_search = RandomizedSearchCV(
    make_pipeline(base_pre(), LogisticRegression(max_iter=2000, class_weight=None)),
    {"logisticregression__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
    n_iter=5, cv=cv, scoring="f1_macro", random_state=SEED, n_jobs=-1)
base_search.fit(X_tr, y_tr)
print(f"baseline CV best params: {base_search.best_params_}")
base_pred = base_search.predict(X_te)
base_res, base_rep = evaluate("baseline", base_pred)

# ---- 5. IMPROVED: class-balanced CV-tuned LightGBM with coverage features ----
# Coverage stats are computed from training labels only (no test leakage).
size_map = s_tr.value_counts().to_dict()
bal_map = pd.Series(y_tr.values).groupby(s_tr.values).mean().to_dict()
Xtr_c = coverage_feats(X_tr, s_tr, size_map, bal_map)
Xte_c = coverage_feats(X_te, s_te, size_map, bal_map)
cov_num = num_cols + ["scen_size", "scen_pos_rate"]
pre_c = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), cov_num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")), cat_cols)])
lgb_pipe = make_pipeline(pre_c, LGBMClassifier(class_weight="balanced", random_state=SEED,
                                               n_estimators=300, verbose=-1, n_jobs=-1))
lgb_search = RandomizedSearchCV(
    lgb_pipe,
    {"lgbmclassifier__num_leaves": [15, 31, 63],
     "lgbmclassifier__learning_rate": [0.03, 0.1],
     "lgbmclassifier__min_child_samples": [20, 50]},
    n_iter=6, cv=cv, scoring="f1_macro", random_state=SEED, n_jobs=-1)
lgb_search.fit(Xtr_c, y_tr)
print(f"improved CV best params: {lgb_search.best_params_}")

# Calibrated ensemble: average probabilities of improved LightGBM and baseline LR,
# weight chosen by CV on training data (out-of-fold probabilities).
from sklearn.base import clone
oof_b = np.zeros(len(X_tr)); oof_l = np.zeros(len(X_tr))
for tr_i, va_i in cv.split(X_tr, y_tr):
    m_b = clone(base_search.best_estimator_).fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i])
    oof_b[va_i] = m_b.predict_proba(X_tr.iloc[va_i])[:, 1]
    m_l = clone(lgb_search.best_estimator_).fit(Xtr_c.iloc[tr_i], y_tr.iloc[tr_i])
    oof_l[va_i] = m_l.predict_proba(Xtr_c.iloc[va_i])[:, 1]
best_w, best_cv = 0.0, -1
for w in np.linspace(0, 1, 11):
    p = w * oof_l + (1 - w) * oof_b
    f = f1_score(y_tr, (p >= 0.5).astype(int), average="macro")
    if f > best_cv:
        best_w, best_cv = w, f
print(f"ensemble weight on LightGBM (CV-selected): {best_w:.1f}")

# Test-set evaluation (single use of the test set for both models)
p_lgb = lgb_search.best_estimator_.predict_proba(Xte_c)[:, 1]
p_base = base_search.best_estimator_.predict_proba(X_te)[:, 1]
p_ens = best_w * p_lgb + (1 - best_w) * p_base
imp_pred = (p_ens >= 0.5).astype(int)
imp_res, imp_rep = evaluate("improved", imp_pred)

# ---- 6. Seed robustness: both models are randomized only through the split-free
# fits; LightGBM subsampling is seeded, so repeat the improved fit with seeds 42-44.
b_scores, i_scores = [], []
for sd in SEEDS:
    m = clone(lgb_search.best_estimator_).set_params(lgbmclassifier__random_state=sd).fit(Xtr_c, y_tr)
    pl = m.predict_proba(Xte_c)[:, 1]
    pe = (best_w * pl + (1 - best_w) * p_base) >= 0.5
    i_scores.append(scenario_report(pe.astype(int), s_te, y_te)["macro_f1"].min())
    mb = clone(base_search.best_estimator_).set_params(logisticregression__random_state=sd).fit(X_tr, y_tr)
    b_scores.append(scenario_report(mb.predict(X_te), s_te, y_te)["macro_f1"].min())

# ---- 7. Charts ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["baseline (LR)", "improved (LGBM+ens)"], [base_res["worst_f1"], imp_res["worst_f1"]], color=["gray", "steelblue"])
ax.set_ylabel("worst-scenario macro-F1 (test)"); ax.set_title("Worst-scenario macro-F1")
fig.tight_layout(); fig.savefig("figures/worst_scenario_f1.png", dpi=120); plt.close(fig)

# ---- 8. Write results ----
n_total = len(X)
results = {
    "question": "Can a class-balanced, cross-validated gradient-boosting model that explicitly models scenario coverage identify under-served scenarios better than a standard per-scenario aggregate baseline on a public stand-in dataset?",
    "method": "Adult income as stand-in; scenarios = education band x sex; stratified 70/30 split; CV tuning on train only; worst-scenario macro-F1 on test.",
    "dataset": {"name": "UCI Adult Income", "source_url": URL, "n_rows": int(n_total),
                "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "same_as_paper": False,
                "baseline_is_papers_method": False},
    "metric_name": "worst-scenario macro-F1 (scenarios with >=50 test rows)",
    "higher_is_better": True,
    "baseline": "Logistic regression (CV-tuned C) reporting per-scenario aggregate macro-F1; no coverage reasoning.",
    "improved": "Class-balanced CV-tuned LightGBM with scenario-size and per-scenario class-balance features, averaged with the baseline via a CV-selected weight.",
    "metrics": {
        "baseline_score": float(base_res["worst_f1"]),
        "improved_score": float(imp_res["worst_f1"]),
        "improvement": float(imp_res["worst_f1"] - base_res["worst_f1"]),
        "baseline_score_std": float(np.std(b_scores)),
        "improved_score_std": float(np.std(i_scores)),
        "baseline_f1": float(base_res["overall_f1"]),
        "improved_f1": float(imp_res["overall_f1"]),
        "baseline_n_below_threshold": base_res["n_below"],
        "improved_n_below_threshold": imp_res["n_below"],
        "n_scenarios_evaluated": int(len(imp_rep)),
        "under_served_threshold": THRESH,
        "ensemble_weight_lgbm": float(best_w),
        "runtime_seconds": float(time.time() - t0)},
    "notes": "Scenarios are a constructed rule on Adult attributes, not a real code-generation benchmark; the paper's reported result is not reproduced. Only one train/test split; seed-based std covers model randomness only. Threshold for under-served scenarios is an arbitrary choice. Small scenarios (<50 test rows) are excluded from the main metric."}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("wrote results.json")
if not np.isfinite(results["metrics"]["improved_score"]):
    sys.exit(1)
