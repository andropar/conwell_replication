# Manually-transferred weights

The DeepNSD-protocol extractor (now vendored at
`src/conwell_replication/_vendor/deepnsd/`) downloads most weights at runtime
via `torch.hub` / `torchvision.datasets.utils.download_url`. The exception
is the **SLIP** family, whose checkpoint files live inside the DeepNSD
source tree itself and are too large to ship through GitHub.

## Required: SLIP weights

DeepNSD's `model_opts/model_code/_options.py` looks up SLIP checkpoints by
filename in `model_opts/model_code/slip_weights/`. We **do not vendor**
those files — they total ~31 GB. Transfer them out-of-band:

    SRC=/home/jroth/dense-retinotopy-func/src/external/DeepNSD/source_code/pressures/model_opts/model_code/slip_weights
    DST=src/conwell_replication/_vendor/deepnsd/model_opts/model_code/slip_weights
    mkdir -p "$DST"
    rsync -avh "$SRC/" "$DST/"

Files expected (used by the 14 SLIP entries in DeepNSD's registry):

- `simclr_small_25ep.pt`, `simclr_base_25ep.pt`, `simclr_large_25ep.pt`
- `clip_small_25ep.pt`, `clip_base_25ep.pt`, `clip_large_25ep.pt`,
  `clip_base_cc12m_35ep.pt`
- `slip_small_25ep.pt`, `slip_base_25ep.pt`, `slip_large_25ep.pt`,
  `slip_base_cc12m_35ep.pt`, `slip_base_50ep.pt`, `slip_base_100ep.pt`
- `slip_base_yfcc15m_25ep.pt`

`.pt` files are gitignored, so they stay out of the repo even if accidentally
copied into the tree.

## Optional: other source-specific caches

Other model sources will populate their own caches under `~/.cache/torch`
or `~/.cache/huggingface` on first run. If you want to seed those from the
current server (avoiding a fresh download on the new server), copy:

| Source     | Cache to copy                                               |
|------------|-------------------------------------------------------------|
| timm       | `~/.cache/huggingface/hub/`                                 |
| torchvision| `~/.cache/torch/hub/checkpoints/`                           |
| dino       | `~/.cache/torch/hub/checkpoints/dino_*.pth`                 |
| midas      | `~/.cache/torch/hub/checkpoints/dpt_*.pt`                   |
| seer       | `_vendor/deepnsd/model_opts/model_code/<seer_*.torch>`      |

## On a fresh server

Download from the VISSL model zoo (Barlow Twins on ImageNet-1K) and rename to
`resnet50_barlowtwins.pth`. See the original VISSL model zoo for the
canonical link.
