# External Model Weights

Most model weights are downloaded at runtime by the underlying model libraries
(`torch.hub`, torchvision, timm, Hugging Face Hub, and related packages). A few
DeepNSD model entries expect checkpoint files to already exist on disk. Those
large files are not committed to this repository.

## SLIP Checkpoints

DeepNSD looks for SLIP-family checkpoints under:

```text
src/conwell_replication/_vendor/deepnsd/model_opts/model_code/slip_weights/
```

Create that directory and copy the checkpoint files into it:

```bash
SRC=/path/to/slip_weights
DST=src/conwell_replication/_vendor/deepnsd/model_opts/model_code/slip_weights
mkdir -p "$DST"
rsync -avh "$SRC/" "$DST/"
```

Expected filenames:

- `simclr_small_25ep.pt`
- `simclr_base_25ep.pt`
- `simclr_large_25ep.pt`
- `clip_small_25ep.pt`
- `clip_base_25ep.pt`
- `clip_large_25ep.pt`
- `clip_base_cc12m_35ep.pt`
- `slip_small_25ep.pt`
- `slip_base_25ep.pt`
- `slip_large_25ep.pt`
- `slip_base_cc12m_35ep.pt`
- `slip_base_50ep.pt`
- `slip_base_100ep.pt`
- `slip_base_yfcc15m_25ep.pt`

Checkpoint files (`*.pt`, `*.pth`, `*.bin`) are ignored by Git.

## Optional Caches

Other model sources populate standard library caches on first use. To avoid
downloading them repeatedly across machines, copy the relevant cache directory
manually:

| Source | Typical cache |
| --- | --- |
| timm | `~/.cache/huggingface/hub/` |
| torchvision | `~/.cache/torch/hub/checkpoints/` |
| DINO | `~/.cache/torch/hub/checkpoints/dino_*.pth` |
| MiDaS | `~/.cache/torch/hub/checkpoints/dpt_*.pt` |

For VISSL Barlow Twins, provide the ImageNet-1K ResNet50 checkpoint expected by
the corresponding DeepNSD model option and name it `resnet50_barlowtwins.pth`.
