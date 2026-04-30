"""Vendored DeepNSD source.

This package mirrors the layout of
``DeepNSD/source_code/pressures/`` (Conwell et al.). The only external code
not pip-installable that the replication depends on, vendored here so the
new server doesn't need a separate DeepNSD checkout.

Layout::

    deepnsd/
      ridge_gcv_mod.py
      model_opts/
        feature_extraction.py    # hook-based, get_all_feature_maps
        feature_reduction.py     # SRP via get_feature_map_srps
        mapping_methods.py       # compare_rdms, score_func, RidgeCVMod helpers
        model_options.py         # registry: get_model_options, get_recommended_transforms
        model_typology.csv       # the trained-model typology Conwell uses
        model_meta/              # per-model accuracy / metadata CSVs
        model_code/
          _options.py            # SLIP / SEER / VICReg / IPCL / BiT / robustness loaders
          custom_extended_models.py
          bit_experts.py
          ipcl_codebase/         # IPCL model definitions
          slip_codebase/         # SLIP model definitions
          vicreg_repo/           # VICReg model definitions
          slip_weights/          # **NOT VENDORED — 31 GB**, transfer separately
                                 # Place SLIP .pt checkpoints here on the new server.

Several DeepNSD modules use absolute imports of the form
``from model_opts.model_code._options import ...``. To keep that working,
we prepend this directory to ``sys.path`` at import time so that
``model_opts`` resolves to the vendored copy.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
if str(_VENDOR_DIR) not in _sys.path:
    _sys.path.insert(0, str(_VENDOR_DIR))
