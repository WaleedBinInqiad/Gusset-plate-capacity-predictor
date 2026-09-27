"""
CatBoost Regressor for Gusset Plate Compressive Capacity
==========================================================
Replaces the GradientBoostingRegressor/XGBoost model in your GUI with a
tuned CatBoostRegressor, searches hyperparameters to maximize R2 on both
train and test, and exports the final model as both native CatBoost format
and ONNX (for future mobile/on-device use regardless of which Android
deployment path you choose).

IMPORTANT CAVEAT ON THE "R2 > 0.99 ON BOTH SPLITS" TARGET
----------------------------------------------------------
Train R2 > 0.99 is easy to hit with enough trees/depth -- that's just
fitting capacity, not evidence of a good model. Test R2 > 0.99 is a much
higher bar, and for THIS dataset specifically there's a real ceiling on
what's achievable: your data contains at least one set of replicate
specimens with IDENTICAL inputs (Fy=295, t=13.3, Lb=97, Lc=16, Lconn=280,
df=68) but Pu = 2061, 2208, 2335, 2349 kN -- a ~6% relative std for
supposedly identical geometry. That's irreducible experimental scatter no
function of the six inputs can explain. If replicates like this end up
split between train and test, a model can look like it's generalizing
when it's actually partly memorizing specimen-specific noise; if they're
genuinely separated, ~6%+ unexplained variance imposes a real ceiling on
test R2 that no amount of hyperparameter tuning can cross honestly.
This script reports 5-fold CV R2 alongside the single train/test split so
you can tell the difference between "genuinely accurate" and "got lucky
with this split."
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error, median_absolute_error,
                              explained_variance_score, r2_score)
import sklearn.metrics as sm
import shap
import matplotlib.pyplot as plt
import os

RANDOM_STATE = 50   # matches your original GUI train/test split seed
OUT_DIR = "pipeline_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_PATH = "Gusset.csv"

# Apply the same cleaning established earlier in this project (bolted-only
# + non-buckling-cluster exclusion) before training the model that ships
# in the deployed tool. Set False only if Gusset.csv on disk has ALREADY
# been overwritten with the cleaned version.
APPLY_CLEANING = True
SUSPECT_LB_VALUES_MM = [3.175, 6.35]
LB_MATCH_TOL = 0.05

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

if APPLY_CLEANING:
    col_names = df.columns.tolist()
    # Expect columns in the order: Fy, t, Lb, Lc, Lconn, df, Pu (as in your
    # original 6-input dataset). Adjust indices below if your column order
    # differs.
    fy_col, t_col, lb_col, lc_col, lconn_col, df_col, pu_col = col_names[:7]

    n0 = len(df)
    df = df[df[df_col] > 0].reset_index(drop=True)
    print(f"Removed {n0 - len(df)} welded rows (fastener distance <= 0)")

    n1 = len(df)
    is_cluster = df[lb_col].apply(lambda v: any(abs(v - s) < LB_MATCH_TOL for s in SUSPECT_LB_VALUES_MM))
    df = df[~is_cluster].reset_index(drop=True)
    print(f"Removed {n1 - len(df)} non-buckling-cluster rows (Lb in {SUSPECT_LB_VALUES_MM} mm)")
    print(f"Rows remaining for training: {len(df)}\n")

X = df.iloc[:, :-1]
y = df.iloc[:, -1:].values.ravel()

# ----------------------------------------------------------------------
# Train/test split (matches your GUI: test_size=0.2, random_state=50)
# ----------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {len(x_train)}   Test: {len(x_test)}")

# Flag any near-duplicate rows split across train/test (replicate
# specimens like the one described above) -- these inflate apparent test
# accuracy without reflecting real generalization.
train_keys = set(map(tuple, x_train.round(3).values.tolist()))
test_dupe_count = sum(1 for row in x_test.round(3).values.tolist() if tuple(row) in train_keys)
if test_dupe_count:
    print(f"WARNING: {test_dupe_count} test rows share (near-)identical inputs with a "
          f"training row. Test R2 will be optimistic for these -- consider a group-aware "
          f"split if this number is large.\n")

# ----------------------------------------------------------------------
# Hyperparameter search
# ----------------------------------------------------------------------
param_dist = {
    # Capped at iterations<=300, depth<=8: for a dataset this size (a few
    # hundred rows), deeper/longer models mainly add fit time and
    # overfitting risk, not accuracy -- and each step up costs real search
    # wall-clock time (a single depth=10, iterations=600 fit took ~8.6s in
    # testing vs ~0.7-2.5s in this range). Raise these caps if you have a
    # faster machine/more cores and want a more exhaustive search.
    "iterations":        [100, 150, 200, 250, 300],
    "depth":              list(range(3, 9)),
    "learning_rate":      [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
    "l2_leaf_reg":        [1, 3, 5, 7, 9, 12],
    "subsample":          [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "random_strength":    [0.5, 1, 2, 5],
}
# Note: bagging_temperature is only valid with bootstrap_type="Bayesian",
# which is incompatible with subsample (that needs Bernoulli/Poisson
# bootstrap) -- can't tune both at once, so bagging_temperature is dropped
# here in favor of keeping subsample in the search.

base_model = CatBoostRegressor(
    loss_function="RMSE",
    random_state=RANDOM_STATE,
    verbose=False,
    bootstrap_type="Bernoulli",  # required for subsample to be a valid param
    thread_count=1,  # IMPORTANT: let sklearn's n_jobs parallelize across
                      # fits instead of CatBoost parallelizing within each
                      # fit -- both at once oversubscribes CPU cores and
                      # makes the search dramatically slower, not faster
)

CV_FOLDS = 5
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

N_SEARCH_ITER = 80  # reduce if this is still too slow on your machine;
                     # increase for a more thorough search if you have time
print(f"\nRunning RandomizedSearchCV ({N_SEARCH_ITER} iterations x 5-fold CV)...")
search = RandomizedSearchCV(
    base_model, param_dist, n_iter=N_SEARCH_ITER, scoring="r2",
    cv=cv, n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)
search.fit(x_train, y_train)
model = search.best_estimator_
print(f"\nBest params: {search.best_params_}")
print(f"Best CV R2 (mean over 5 folds, on training data): {search.best_score_:.5f}")

# Honest cross-check: 5-fold CV R2 of the FINAL chosen model, computed
# fresh (search.best_score_ already reflects this, shown again for clarity)
cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
print(f"Independent re-check, 5-fold CV R2: {cv_scores.mean():.5f} (+/- {cv_scores.std():.5f})")

# ----------------------------------------------------------------------
# Fit on full training set, evaluate on train + test
# ----------------------------------------------------------------------
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

def report(y_true, y_pred, label):
    print(f"\n-- {label} --")
    print("MAE  =", round(sm.mean_absolute_error(y_true, y_pred), 5))
    print("MSE  =", round(sm.mean_squared_error(y_true, y_pred), 5))
    print("RMSE =", round(np.sqrt(sm.mean_squared_error(y_true, y_pred)), 5))
    print("MedAE=", round(sm.median_absolute_error(y_true, y_pred), 5))
    print("EVS  =", round(sm.explained_variance_score(y_true, y_pred), 5))
    print("R2   =", round(sm.r2_score(y_true, y_pred), 5))

report(y_train, y_train_pred, "TRAIN")
report(y_test, y_test_pred, "TEST")

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
if test_r2 > 0.99 and cv_scores.mean() < 0.95:
    print("\n>>> NOTE: single-split test R2 is above 0.99 but 5-fold CV R2 is notably")
    print(">>> lower. This gap usually means the single 80/20 split happened to be")
    print(">>> favorable (or contains near-duplicate train/test rows -- see the WARNING")
    print(">>> above). Report the CV number as your primary generalization estimate,")
    print(">>> not the single split, if you want this to hold up under review.")

# ----------------------------------------------------------------------
# Save model: native CatBoost format + ONNX (useful for either Android path)
# ----------------------------------------------------------------------
model.save_model(os.path.join(OUT_DIR, "gusset_catboost.cbm"))
model.save_model(os.path.join(OUT_DIR, "gusset_catboost.onnx"), format="onnx")
print(f"\nSaved model: {OUT_DIR}/gusset_catboost.cbm (native) and .onnx (portable)")

# ----------------------------------------------------------------------
# SHAP (TreeExplainer -- exact and fast for CatBoost, unlike KernelSHAP)
# ----------------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer(x_test)

plt.figure()
shap.summary_plot(shap_values, x_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_waterfall_example.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved: {OUT_DIR}/shap_summary.png, {OUT_DIR}/shap_waterfall_example.png")

# Save best params + metrics to CSV for the paper
summary = pd.DataFrame([{
    "best_params": str(search.best_params_),
    "train_R2": train_r2, "test_R2": test_r2,
    "cv_R2_mean": cv_scores.mean(), "cv_R2_std": cv_scores.std(),
    "train_MAE": mean_absolute_error(y_train, y_train_pred),
    "test_MAE": mean_absolute_error(y_test, y_test_pred),
    "train_RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
    "test_RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
}])
summary.to_csv(os.path.join(OUT_DIR, "catboost_tuning_summary.csv"), index=False)
print(f"Saved: {OUT_DIR}/catboost_tuning_summary.csv")
