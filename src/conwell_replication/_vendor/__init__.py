"""Vendored third-party code.

Currently a single subpackage:

- :mod:`conwell_replication._vendor.deepnsd` — Conwell's DeepNSD/pressures
  tree (model_opts, ridge_gcv_mod, model_code/{slip,vicreg,ipcl}_codebase).
  Used by both the extractor (``get_model_options``, ``get_all_feature_maps``,
  ``get_recommended_transforms``) and by the evaluators (``compare_rdms``,
  ``score_func``, ``RidgeCVMod``).
"""

from . import deepnsd  # noqa: F401  — runs the sys.path prepend so ``model_opts`` is importable
