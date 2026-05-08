#!/bin/bash
# Install the cheap pip extras into the Raven conda env.
# Run on Raven login (not Viper).
#
# Bucket 1 from our model-source audit — covers ~39 additional Conwell models
# (clip 8 + openclip 4 + taskonomy 25 + vqgan 1 + robustness 1) without
# pulling in fragile heavyweights (detectron2, vissl, tensorflow). Each is
# pure-python, doesn't conflict with our CUDA-13 torch wheel.

set -eo pipefail

ENV=/raven/u/rothj/conda-envs/conwell_replication
REPO=/raven/u/rothj/conwell_replication

echo "=== Installing cheap extras: clip openclip taskonomy vqgan robustness"
"${ENV}/bin/pip" install -e "${REPO}[clip,openclip,taskonomy,vqgan,robustness]"

echo
echo "=== Verifying each extra imports:"
"${ENV}/bin/python" - <<'PY'
import importlib
mods = {
    "clip":              "clip",            # OpenAI CLIP
    "open_clip_torch":   "open_clip",       # MLab's OpenCLIP
    "visualpriors":      "visualpriors",    # Taskonomy
    "taming-transformers":"taming",         # VQGAN
    "robustness":        "robustness",
}
ok, bad = [], []
for pkg, modname in mods.items():
    try:
        importlib.import_module(modname)
        ok.append(pkg)
    except Exception as e:
        bad.append((pkg, str(e).splitlines()[0]))
print("  OK:    ", ", ".join(ok))
if bad:
    print("  BROKEN:")
    for p, e in bad:
        print(f"    {p}: {e}")
PY

echo
echo "=== Done."
