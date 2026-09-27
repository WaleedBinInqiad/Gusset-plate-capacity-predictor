"""
Export a tuned CatBoostRegressor + real SHAP explanations into a single
compact JSON bundle that a fully self-contained, offline-capable web page
can load -- no Python backend, no server, works as an installable Android
PWA.

WHY THIS DESIGN (read before changing it)
------------------------------------------
Prediction: CatBoost's default trees are "oblivious" (symmetric) -- every
tree is just D binary feature comparisons applied in a fixed order, with a
lookup into 2^D leaf values. That's simple and exact to re-implement in
plain JavaScript, so the web app's predicted number matches Python
bit-for-bit. This script exports that tree structure directly.

Explanation: A from-scratch TreeSHAP reimplementation in JS was tried and
validated against shap.TreeExplainer -- it reproduced the total predicted
value exactly, but individual per-feature attributions were off by
5-10% in several cases (likely a subtlety in how CatBoost's own SHAP
implementation weights coverage internally that isn't fully documented).
Rather than ship an explanation that LOOKS like SHAP but is quietly wrong,
this script instead precomputes REAL shap.TreeExplainer values (guaranteed
correct, since it's the actual library) across a dense sample of the
input space, and the web app does nearest-neighbor lookup/interpolation
at runtime. This is an approximation between sampled points, but every
number it's built from is a genuine SHAP value, not a reimplementation.

Run this AFTER gusset_catboost_tuning.py has produced
pipeline_outputs/gusset_catboost.cbm.
"""

import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import shap
import os

OUT_DIR = "pipeline_outputs"
MODEL_PATH = os.path.join(OUT_DIR, "gusset_catboost.cbm")
DATA_PATH = "Gusset.csv"
N_SAMPLES = 4000          # size of the precomputed SHAP lookup table
RANDOM_STATE = 42

FEATURE_LABELS = [
    "Yield Strength (MPa)", "Plate Thickness (mm)", "Buckling Length (mm)",
    "Cantilever Length (mm)", "Connection Length (mm)", "Fastener Distance (mm)",
]

# ----------------------------------------------------------------------
# Load model + data
# ----------------------------------------------------------------------
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
X = df.iloc[:, :-1]
y = df.iloc[:, -1:].values.ravel()
feature_names = list(X.columns)
n_feat = len(feature_names)

# ----------------------------------------------------------------------
# 1. Export the tree structure for exact in-browser prediction
# ----------------------------------------------------------------------
tmp_json_path = os.path.join(OUT_DIR, "_model_dump.json")
model.save_model(tmp_json_path, format="json")
raw = json.load(open(tmp_json_path))
scale, bias = raw["scale_and_bias"]
bias = bias[0] if isinstance(bias, list) else bias

trees_compact = []
for t in raw["oblivious_trees"]:
    trees_compact.append({
        "f": [s["float_feature_index"] for s in t["splits"]],  # feature index per depth
        "b": [s["border"] for s in t["splits"]],                # border per depth
        "v": t["leaf_values"],                                  # leaf values, length 2^D
    })

model_bundle = {
    "scale": scale,
    "bias": bias,
    "trees": trees_compact,
    "feature_names": feature_names,
    "feature_labels": FEATURE_LABELS,
}

# Sanity check: reimplemented prediction must match model.predict() exactly
def js_style_predict(x, bundle):
    total = 0.0
    for t in bundle["trees"]:
        idx = 0
        for d, (fi, border) in enumerate(zip(t["f"], t["b"])):
            if x[fi] > border:
                idx |= (1 << d)
        total += t["v"][idx]
    return bundle["bias"] + bundle["scale"] * total

check_rows = X.sample(min(20, len(X)), random_state=1)
real_preds = model.predict(check_rows)
reimpl_preds = [js_style_predict(row.values, model_bundle) for _, row in check_rows.iterrows()]
max_err = np.max(np.abs(np.array(real_preds) - np.array(reimpl_preds)))
print(f"Reimplemented-inference sanity check: max abs error vs model.predict() = {max_err:.6f}")
assert max_err < 1e-4, "Tree reimplementation does not match CatBoost predictions -- do not ship this bundle."

# ----------------------------------------------------------------------
# 2. Precompute REAL SHAP values across a dense sample of the input space
# ----------------------------------------------------------------------
feat_min = X.min().values
feat_max = X.max().values
# widen slightly so the table also covers modest extrapolation
pad = (feat_max - feat_min) * 0.05
lo = np.maximum(feat_min - pad, 0.0)  # these are all physical lengths/strengths -- never negative
hi = feat_max + pad

rng = np.random.default_rng(RANDOM_STATE)
sample = rng.uniform(lo, hi, size=(N_SAMPLES, n_feat))
sample_df = pd.DataFrame(sample, columns=feature_names)

print(f"Computing SHAP values for {N_SAMPLES} sampled points (this may take a bit)...")
explainer = shap.TreeExplainer(model)
shap_vals = explainer(sample_df).values          # (N_SAMPLES, n_feat)
preds = model.predict(sample_df)                  # (N_SAMPLES,)
base_value = float(np.asarray(explainer.expected_value).ravel()[0])

# Normalize inputs to [0,1] per feature for fast nearest-neighbor search in JS
scale_lo = lo.tolist()
scale_range = (hi - lo).tolist()
sample_norm = ((sample - lo) / (hi - lo)).astype(np.float32)

lookup_bundle = {
    "feat_lo": scale_lo,
    "feat_range": scale_range,
    "base_value": base_value,
    "inputs_norm": np.round(sample_norm, 4).tolist(),
    "shap": np.round(shap_vals, 3).tolist(),
    "pred": np.round(preds, 3).tolist(),
}

full_bundle = {"model": model_bundle, "lookup": lookup_bundle}

# Emit as a plain JS file (const assignment), not a .json file loaded via
# fetch() -- <script src="model_data.js"> works when the HTML is opened
# directly from disk (file://); fetch() of a local JSON file is blocked by
# the browser's CORS policy in that case, which would break the app for
# anyone who just double-clicks index.html instead of hosting it.
out_path = os.path.join(OUT_DIR, "model_data.js")
with open(out_path, "w") as f:
    f.write("// Auto-generated by export_for_webapp.py -- do not edit by hand.\n")
    f.write("const GUSSET_MODEL_BUNDLE = ")
    json.dump(full_bundle, f)
    f.write(";\n")

size_kb = os.path.getsize(out_path) / 1024
print(f"Saved bundle: {out_path} ({size_kb:.0f} KB)")
print(f"Base value (SHAP expected output): {base_value:.3f}")
print("Feature ranges used for the lookup table:")
for name, l, h in zip(feature_names, lo, hi):
    print(f"  {name}: [{l:.2f}, {h:.2f}]")
