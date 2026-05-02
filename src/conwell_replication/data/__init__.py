"""Data layer: LAION-fMRI benchmark adapter + stimulus-pool builder.

Train/test splits are not bundled here — they live in
:mod:`laion_fmri.splits` (the upstream package). Import from there:

    from laion_fmri.splits import (
        list_pools, list_splits,
        load_split, get_train_test_ids, get_split_masks,
    )
"""

from .benchmark import LAIONBenchmark

__all__ = ["LAIONBenchmark"]
