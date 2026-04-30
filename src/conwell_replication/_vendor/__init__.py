"""Vendored third-party code.

- ridge_gcv_mod, mapping_methods, feature_reduction:
    Subset of DeepNSD/source_code/pressures (Conwell, by way of the Hebart Lab
    fork at /home/jroth/dense-retinotopy-func/src/external/DeepNSD/). Vendored
    verbatim except for an import guard in feature_reduction.py.

- universal_extractor:
    cstims/feature_extraction/universal_extractor.py from
    /home/jroth/rsa_based_selection/. Vendored verbatim — delegates to
    deepjuice for most models and to a small set of custom loaders for the
    rest. The custom loaders depend on weight files in `resources/weights/`
    (see the ``RESOURCES_DIR`` constant in that file; we override it on
    import below).
"""

from pathlib import Path as _Path
from . import universal_extractor as _ue

# Point the universal extractor at our packaged resources directory rather
# than the original cstims layout (`cstims/data/resources`). Custom-weight
# files (e.g. resnet50_barlowtwins.pth) should live under
# resources/weights/ in this repo.
_ue.RESOURCES_DIR = _Path(__file__).resolve().parents[3] / "resources" / "weights"
