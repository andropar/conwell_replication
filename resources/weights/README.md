# Custom model weights

This directory hosts weight files for the small set of models that are loaded
through `conwell_replication._vendor.universal_extractor.get_custom_model`
rather than through deepjuice.

Among the **152 curated models**, only one needs a manually-supplied weight
file:

| Model                          | Required file                  |
|--------------------------------|--------------------------------|
| `vissl_resnet50_barlowtwins`   | `resnet50_barlowtwins.pth`     |

The others routed through `get_custom_model` self-download:

- `cornet_s` — fetched via the `cornet` package
- `dinov3-vitl16-pretrain-lvd1689m` — fetched via 🤗 `transformers`

## On the development host (current server)

The original file lives at:

    /home/jroth/rsa_based_selection/data/resources/resnet50_barlowtwins.pth   (90 MB)

Copy it into this directory before the first extraction run:

    cp /home/jroth/rsa_based_selection/data/resources/resnet50_barlowtwins.pth \
       resources/weights/

`*.pth` files are gitignored, so the weight stays out of the repo.

## On a fresh server

Download from the VISSL model zoo (Barlow Twins on ImageNet-1K) and rename to
`resnet50_barlowtwins.pth`. See the original VISSL model zoo for the
canonical link.
