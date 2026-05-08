# LAION-fMRI Conwell Fig. 2-4 Replication Report

This report summarizes the completed OTC analyses for the Conwell-style controlled comparisons: architecture variation (Fig. 2), task variation (Fig. 3), and input-diet variation (Fig. 4). The internal metric named `wrsa` is the voxel-encoding RSA metric corresponding most closely to Conwell's veRSA, so it is labeled here as wRSA/veRSA.

The LAION-fMRI analyses use 5 subjects and 117 plotted models after dropping the known `efficientnet_b1_classification` coverage issue. Conwell's paper used the NSD shared-image setting with 4 subjects and a larger model survey, so the most important comparison is the direction and relative size of controlled effects rather than exact equality of absolute scores.

## Supporting Tables

- [Score summary CSV](laion_fmri_conwell_fig2_4_score_summary.csv)
- [Effect comparison CSV](laion_fmri_conwell_fig2_4_effects.csv)
- [Taskonomy endpoint CSV](laion_fmri_conwell_fig2_4_taskonomy_endpoints.csv)

## Table-Derived Summary Figures

**Overall score summary** ([PDF](figures/overview_score_summary.pdf))

<img src="figures/overview_score_summary.png" alt="Overall score summary" width="900">

**Shared-image versus subject-image pool comparison** ([PDF](figures/shared_vs_subject_pool.pdf))

<img src="figures/shared_vs_subject_pool.png" alt="Shared versus subject image pool paired differences" width="900">

## Initial Reading

- Absolute LAION-fMRI wRSA/veRSA scores are generally below the headline Conwell NSD values, especially for split-half, cluster, and OOD tests. The strongest LAION-fMRI aggregates are random and tau splits, while cluster and OOD splits are visibly harder.
- The cRSA architecture effect consistently preserves the Conwell direction: transformers are below CNNs. In wRSA/veRSA, however, the LAION-fMRI estimates often flip slightly positive for transformers, so the architecture conclusion is not a literal replication in the voxel-encoding metric.
- The contrastive self-supervised advantage is robust in the random, cluster, tau, and plain split-half analyses, broadly matching Conwell's qualitative result.
- The SLIP comparison mostly matches Conwell's qualitative pattern in the split variants where CLIP is below SimCLR and SLIP is close to SimCLR. The OOD-inclusive split-half result should be read more cautiously because its score scale differs from the other split families.
- The ImageNet21K comparison remains small, consistent with Conwell's conclusion that the larger ImageNet21K diet does not clearly improve OTC predictivity. Some LAION-fMRI wRSA/veRSA estimates are slightly negative.
- The IPCL input-diet result is only partial: Places and VGGFace2 are often below ImageNet in wRSA/veRSA, but the cRSA VGGFace2 effect is not consistently Conwell-like and OOD results can invert. This is the main controlled input-diet comparison to inspect manually in Fig. 4.

## Caveats

- These analyses use the current OTC ROI and the plotted 117-model subset. Conwell's paper reports NSD OTC results over a larger model survey and 4 subjects.
- OOD sections use the aggregate `ood_type=all` rows from the current evaluator. They do not show the nine OOD types separately.
- The language-alignment table compares CLIP and SLIP to SimCLR using the implemented Conwell-style fixed-effect test. It does not include the model-size interaction discussed in the paper narrative.
- SRPR is included descriptively because it is useful for this project, but it is not one of the Fig. 2-4 Conwell paper metrics.

## Conwell Reference Statistics Used Here

- Architecture: transformers were slightly below CNNs in Conwell (cRSA beta -0.04; veRSA beta -0.01).
- Taskonomy: Conwell reported low autoencoding scores and higher object-classification scores, with Taskonomy still below ImageNet-trained ResNet50.
- Self-supervised learning: Conwell reported an instance-level contrastive advantage of about 0.06 in cRSA and 0.09 in veRSA.
- SLIP: Conwell reported CLIP below SimCLR by about 0.05 in cRSA and 0.02 in veRSA, with no substantial SLIP effect.
- Input diet: Conwell reported little/no ImageNet21K advantage over ImageNet1K, and IPCL diets ordered roughly ImageNet > OpenImages > Places > VGGFace2.

The per-section summary figures below visualize the tables that are printed in full in the appendix.

---

## Odd/Even Split-Half

Closest direct replication of Conwell's shared-image split-half setup.

### Table-Derived Figures

**Score summary** ([PDF](figures/splithalf_score_summary.pdf))

<img src="figures/splithalf_score_summary.png" alt="splithalf score summary" width="900">

**Controlled-effect comparison** ([PDF](figures/splithalf_controlled_effects.pdf))

<img src="figures/splithalf_controlled_effects.png" alt="splithalf effects" width="900">

**Taskonomy endpoint comparison** ([PDF](figures/splithalf_taskonomy_endpoints.pdf))

<img src="figures/splithalf_taskonomy_endpoints.png" alt="splithalf Taskonomy endpoints" width="900">

### Conwell-Style Model Figures

#### split-half

**Fig. 2: architecture variation** ([PDF](../figures/splithalf_results/fig2_architecture_variation.pdf))

<img src="../figures/splithalf_results/fig2_architecture_variation.png" alt="split-half Fig. 2: architecture variation" width="900">

**Fig. 3: task variation** ([PDF](../figures/splithalf_results/fig3_task_variation.pdf))

<img src="../figures/splithalf_results/fig3_task_variation.png" alt="split-half Fig. 3: task variation" width="900">

**Fig. 4: input variation** ([PDF](../figures/splithalf_results/fig4_input_variation.pdf))

<img src="../figures/splithalf_results/fig4_input_variation.png" alt="split-half Fig. 4: input variation" width="900">

---

## Odd/Even Split-Half + OOD

Existing OOD-inclusive split-half analysis.

### Table-Derived Figures

**Score summary** ([PDF](figures/splithalf_ood_score_summary.pdf))

<img src="figures/splithalf_ood_score_summary.png" alt="splithalf_ood score summary" width="900">

**Controlled-effect comparison** ([PDF](figures/splithalf_ood_controlled_effects.pdf))

<img src="figures/splithalf_ood_controlled_effects.png" alt="splithalf_ood effects" width="900">

**Taskonomy endpoint comparison** ([PDF](figures/splithalf_ood_taskonomy_endpoints.pdf))

<img src="figures/splithalf_ood_taskonomy_endpoints.png" alt="splithalf_ood Taskonomy endpoints" width="900">

### Conwell-Style Model Figures

#### split-half + OOD

**Fig. 2: architecture variation** ([PDF](../figures/splithalf_results_ood/fig2_architecture_variation.pdf))

<img src="../figures/splithalf_results_ood/fig2_architecture_variation.png" alt="split-half + OOD Fig. 2: architecture variation" width="900">

**Fig. 3: task variation** ([PDF](../figures/splithalf_results_ood/fig3_task_variation.pdf))

<img src="../figures/splithalf_results_ood/fig3_task_variation.png" alt="split-half + OOD Fig. 3: task variation" width="900">

**Fig. 4: input variation** ([PDF](../figures/splithalf_results_ood/fig4_input_variation.pdf))

<img src="../figures/splithalf_results_ood/fig4_input_variation.png" alt="split-half + OOD Fig. 4: input variation" width="900">

---

## Random Splits, Averaged Across 5 Repeats

Five random train/test splits averaged before best-layer selection.

### Table-Derived Figures

**Score summary** ([PDF](figures/random_agg_score_summary.pdf))

<img src="figures/random_agg_score_summary.png" alt="random_agg score summary" width="900">

**Controlled-effect comparison** ([PDF](figures/random_agg_controlled_effects.pdf))

<img src="figures/random_agg_controlled_effects.png" alt="random_agg effects" width="900">

**Taskonomy endpoint comparison** ([PDF](figures/random_agg_taskonomy_endpoints.pdf))

<img src="figures/random_agg_taskonomy_endpoints.png" alt="random_agg Taskonomy endpoints" width="900">

### Conwell-Style Model Figures

#### shared-image pool

**Fig. 2: architecture variation** ([PDF](../figures/min_nn_results/shared_random_agg/fig2_architecture_variation.pdf))

<img src="../figures/min_nn_results/shared_random_agg/fig2_architecture_variation.png" alt="shared-image pool Fig. 2: architecture variation" width="900">

**Fig. 3: task variation** ([PDF](../figures/min_nn_results/shared_random_agg/fig3_task_variation.pdf))

<img src="../figures/min_nn_results/shared_random_agg/fig3_task_variation.png" alt="shared-image pool Fig. 3: task variation" width="900">

**Fig. 4: input variation** ([PDF](../figures/min_nn_results/shared_random_agg/fig4_input_variation.pdf))

<img src="../figures/min_nn_results/shared_random_agg/fig4_input_variation.png" alt="shared-image pool Fig. 4: input variation" width="900">

#### subject-image pool

**Fig. 2: architecture variation** ([PDF](../figures/min_nn_results/subject_random_agg/fig2_architecture_variation.pdf))

<img src="../figures/min_nn_results/subject_random_agg/fig2_architecture_variation.png" alt="subject-image pool Fig. 2: architecture variation" width="900">

**Fig. 3: task variation** ([PDF](../figures/min_nn_results/subject_random_agg/fig3_task_variation.pdf))

<img src="../figures/min_nn_results/subject_random_agg/fig3_task_variation.png" alt="subject-image pool Fig. 3: task variation" width="900">

**Fig. 4: input variation** ([PDF](../figures/min_nn_results/subject_random_agg/fig4_input_variation.pdf))

<img src="../figures/min_nn_results/subject_random_agg/fig4_input_variation.png" alt="subject-image pool Fig. 4: input variation" width="900">

---

## Cluster Splits, Averaged Across 5 Repeats

Five k-means cluster splits averaged before best-layer selection.

### Table-Derived Figures

**Score summary** ([PDF](figures/cluster_k5_agg_score_summary.pdf))

<img src="figures/cluster_k5_agg_score_summary.png" alt="cluster_k5_agg score summary" width="900">

**Controlled-effect comparison** ([PDF](figures/cluster_k5_agg_controlled_effects.pdf))

<img src="figures/cluster_k5_agg_controlled_effects.png" alt="cluster_k5_agg effects" width="900">

**Taskonomy endpoint comparison** ([PDF](figures/cluster_k5_agg_taskonomy_endpoints.pdf))

<img src="figures/cluster_k5_agg_taskonomy_endpoints.png" alt="cluster_k5_agg Taskonomy endpoints" width="900">

### Conwell-Style Model Figures

#### shared-image pool

**Fig. 2: architecture variation** ([PDF](../figures/min_nn_results/shared_cluster_k5_agg/fig2_architecture_variation.pdf))

<img src="../figures/min_nn_results/shared_cluster_k5_agg/fig2_architecture_variation.png" alt="shared-image pool Fig. 2: architecture variation" width="900">

**Fig. 3: task variation** ([PDF](../figures/min_nn_results/shared_cluster_k5_agg/fig3_task_variation.pdf))

<img src="../figures/min_nn_results/shared_cluster_k5_agg/fig3_task_variation.png" alt="shared-image pool Fig. 3: task variation" width="900">

**Fig. 4: input variation** ([PDF](../figures/min_nn_results/shared_cluster_k5_agg/fig4_input_variation.pdf))

<img src="../figures/min_nn_results/shared_cluster_k5_agg/fig4_input_variation.png" alt="shared-image pool Fig. 4: input variation" width="900">

#### subject-image pool

**Fig. 2: architecture variation** ([PDF](../figures/min_nn_results/subject_cluster_k5_agg/fig2_architecture_variation.pdf))

<img src="../figures/min_nn_results/subject_cluster_k5_agg/fig2_architecture_variation.png" alt="subject-image pool Fig. 2: architecture variation" width="900">

**Fig. 3: task variation** ([PDF](../figures/min_nn_results/subject_cluster_k5_agg/fig3_task_variation.pdf))

<img src="../figures/min_nn_results/subject_cluster_k5_agg/fig3_task_variation.png" alt="subject-image pool Fig. 3: task variation" width="900">

**Fig. 4: input variation** ([PDF](../figures/min_nn_results/subject_cluster_k5_agg/fig4_input_variation.pdf))

<img src="../figures/min_nn_results/subject_cluster_k5_agg/fig4_input_variation.png" alt="subject-image pool Fig. 4: input variation" width="900">

---

## Min-NN Threshold Split

Nearest-neighbor threshold split.

### Table-Derived Figures

**Score summary** ([PDF](figures/tau_score_summary.pdf))

<img src="figures/tau_score_summary.png" alt="tau score summary" width="900">

**Controlled-effect comparison** ([PDF](figures/tau_controlled_effects.pdf))

<img src="figures/tau_controlled_effects.png" alt="tau effects" width="900">

**Taskonomy endpoint comparison** ([PDF](figures/tau_taskonomy_endpoints.pdf))

<img src="figures/tau_taskonomy_endpoints.png" alt="tau Taskonomy endpoints" width="900">

### Conwell-Style Model Figures

#### shared-image pool

**Fig. 2: architecture variation** ([PDF](../figures/min_nn_results/shared_tau/fig2_architecture_variation.pdf))

<img src="../figures/min_nn_results/shared_tau/fig2_architecture_variation.png" alt="shared-image pool Fig. 2: architecture variation" width="900">

**Fig. 3: task variation** ([PDF](../figures/min_nn_results/shared_tau/fig3_task_variation.pdf))

<img src="../figures/min_nn_results/shared_tau/fig3_task_variation.png" alt="shared-image pool Fig. 3: task variation" width="900">

**Fig. 4: input variation** ([PDF](../figures/min_nn_results/shared_tau/fig4_input_variation.pdf))

<img src="../figures/min_nn_results/shared_tau/fig4_input_variation.png" alt="shared-image pool Fig. 4: input variation" width="900">

#### subject-image pool

**Fig. 2: architecture variation** ([PDF](../figures/min_nn_results/subject_tau/fig2_architecture_variation.pdf))

<img src="../figures/min_nn_results/subject_tau/fig2_architecture_variation.png" alt="subject-image pool Fig. 2: architecture variation" width="900">

**Fig. 3: task variation** ([PDF](../figures/min_nn_results/subject_tau/fig3_task_variation.pdf))

<img src="../figures/min_nn_results/subject_tau/fig3_task_variation.png" alt="subject-image pool Fig. 3: task variation" width="900">

**Fig. 4: input variation** ([PDF](../figures/min_nn_results/subject_tau/fig4_input_variation.pdf))

<img src="../figures/min_nn_results/subject_tau/fig4_input_variation.png" alt="subject-image pool Fig. 4: input variation" width="900">

---

## OOD Test Split

Model fit once and evaluated on the aggregate OOD test set.

### Table-Derived Figures

**Score summary** ([PDF](figures/ood_score_summary.pdf))

<img src="figures/ood_score_summary.png" alt="ood score summary" width="900">

**Controlled-effect comparison** ([PDF](figures/ood_controlled_effects.pdf))

<img src="figures/ood_controlled_effects.png" alt="ood effects" width="900">

**Taskonomy endpoint comparison** ([PDF](figures/ood_taskonomy_endpoints.pdf))

<img src="figures/ood_taskonomy_endpoints.png" alt="ood Taskonomy endpoints" width="900">

### Conwell-Style Model Figures

#### shared-image pool

**Fig. 2: architecture variation** ([PDF](../figures/min_nn_results/shared_ood/fig2_architecture_variation.pdf))

<img src="../figures/min_nn_results/shared_ood/fig2_architecture_variation.png" alt="shared-image pool Fig. 2: architecture variation" width="900">

**Fig. 3: task variation** ([PDF](../figures/min_nn_results/shared_ood/fig3_task_variation.pdf))

<img src="../figures/min_nn_results/shared_ood/fig3_task_variation.png" alt="shared-image pool Fig. 3: task variation" width="900">

**Fig. 4: input variation** ([PDF](../figures/min_nn_results/shared_ood/fig4_input_variation.pdf))

<img src="../figures/min_nn_results/shared_ood/fig4_input_variation.png" alt="shared-image pool Fig. 4: input variation" width="900">

#### subject-image pool

**Fig. 2: architecture variation** ([PDF](../figures/min_nn_results/subject_ood/fig2_architecture_variation.pdf))

<img src="../figures/min_nn_results/subject_ood/fig2_architecture_variation.png" alt="subject-image pool Fig. 2: architecture variation" width="900">

**Fig. 3: task variation** ([PDF](../figures/min_nn_results/subject_ood/fig3_task_variation.pdf))

<img src="../figures/min_nn_results/subject_ood/fig3_task_variation.png" alt="subject-image pool Fig. 3: task variation" width="900">

**Fig. 4: input variation** ([PDF](../figures/min_nn_results/subject_ood/fig4_input_variation.pdf))

<img src="../figures/min_nn_results/subject_ood/fig4_input_variation.png" alt="subject-image pool Fig. 4: input variation" width="900">

---

## Appendix: Detailed Tables

### Overall Score Summary

| split | variant | metric | mean | median | min | max | subjects | models |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| splithalf | split-half | cRSA | 0.166 | 0.160 | -0.059 | 0.352 | 5 | 117 |
| splithalf | split-half | wRSA/veRSA | 0.383 | 0.430 | -0.088 | 0.588 | 5 | 117 |
| splithalf | split-half | SRPR | 0.376 | 0.417 | 0.079 | 0.481 | 5 | 117 |
| splithalf_ood | split-half + OOD | cRSA | 0.239 | 0.253 | -0.229 | 0.547 | 5 | 117 |
| splithalf_ood | split-half + OOD | wRSA/veRSA | 0.506 | 0.527 | -0.028 | 0.648 | 5 | 117 |
| splithalf_ood | split-half + OOD | SRPR | 0.387 | 0.422 | 0.091 | 0.475 | 5 | 117 |
| random_agg | shared-image pool | cRSA | 0.192 | 0.186 | 0.010 | 0.412 | 5 | 117 |
| random_agg | shared-image pool | wRSA/veRSA | 0.423 | 0.471 | -0.054 | 0.621 | 5 | 117 |
| random_agg | shared-image pool | SRPR | 0.399 | 0.437 | 0.111 | 0.499 | 5 | 117 |
| random_agg | subject-image pool | cRSA | 0.174 | 0.173 | 0.014 | 0.346 | 5 | 117 |
| random_agg | subject-image pool | wRSA/veRSA | 0.422 | 0.454 | -0.010 | 0.625 | 5 | 117 |
| random_agg | subject-image pool | SRPR | 0.366 | 0.399 | 0.133 | 0.449 | 5 | 117 |
| cluster_k5_agg | shared-image pool | cRSA | 0.171 | 0.167 | -0.036 | 0.416 | 5 | 117 |
| cluster_k5_agg | shared-image pool | wRSA/veRSA | 0.318 | 0.338 | -0.074 | 0.498 | 5 | 117 |
| cluster_k5_agg | shared-image pool | SRPR | 0.304 | 0.337 | 0.072 | 0.392 | 5 | 117 |
| cluster_k5_agg | subject-image pool | cRSA | 0.169 | 0.168 | 0.003 | 0.342 | 5 | 117 |
| cluster_k5_agg | subject-image pool | wRSA/veRSA | 0.344 | 0.356 | -0.018 | 0.522 | 5 | 117 |
| cluster_k5_agg | subject-image pool | SRPR | 0.295 | 0.325 | 0.082 | 0.376 | 5 | 117 |
| tau | shared-image pool | cRSA | 0.172 | 0.174 | -0.078 | 0.388 | 5 | 117 |
| tau | shared-image pool | wRSA/veRSA | 0.386 | 0.405 | -0.050 | 0.645 | 5 | 117 |
| tau | shared-image pool | SRPR | 0.384 | 0.421 | 0.097 | 0.488 | 5 | 117 |
| tau | subject-image pool | cRSA | 0.171 | 0.169 | 0.005 | 0.330 | 5 | 117 |
| tau | subject-image pool | wRSA/veRSA | 0.415 | 0.447 | -0.021 | 0.632 | 5 | 117 |
| tau | subject-image pool | SRPR | 0.367 | 0.400 | 0.123 | 0.446 | 5 | 117 |
| ood | shared-image pool | cRSA | 0.081 | 0.077 | -0.329 | 0.639 | 5 | 117 |
| ood | shared-image pool | wRSA/veRSA | 0.217 | 0.226 | -0.040 | 0.604 | 5 | 117 |
| ood | shared-image pool | SRPR | 0.319 | 0.347 | -0.037 | 0.415 | 5 | 117 |
| ood | subject-image pool | cRSA | 0.073 | 0.063 | -0.329 | 0.639 | 5 | 117 |
| ood | subject-image pool | wRSA/veRSA | 0.230 | 0.231 | -0.018 | 0.620 | 5 | 117 |
| ood | subject-image pool | SRPR | 0.343 | 0.366 | -0.058 | 0.433 | 5 | 117 |

### Odd/Even Split-Half

#### Score Summary

| variant | metric | n | subjects | models | mean | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split-half | cRSA | 585 | 5 | 117 | 0.166 | 0.160 | -0.059 | 0.352 |
| split-half | wRSA/veRSA | 585 | 5 | 117 | 0.383 | 0.430 | -0.088 | 0.588 |
| split-half | SRPR | 585 | 5 | 117 | 0.376 | 0.417 | 0.079 | 0.481 |

#### Controlled-Effect Comparison

| family | effect | metric | Conwell paper | split-half |
| --- | --- | --- | --- | --- |
| Architecture | Transformer - CNN | cRSA | -0.04 [-0.05, -0.03], p < 0.001 | -0.073 [-0.086, -0.060], p <0.001 |
| Architecture | Transformer - CNN | wRSA/veRSA | -0.01 [-0.02, -0.00], p < 0.001 | +0.013 [+0.004, +0.023], p 0.008 |
| Self-supervised | Contrastive - Non-Contrastive | cRSA | +0.06 contrastive advantage, p < 0.001 | +0.101 [+0.078, +0.123], p <0.001 |
| Self-supervised | Contrastive - Non-Contrastive | wRSA/veRSA | +0.09 contrastive advantage, p < 0.001 | +0.083 [+0.069, +0.097], p <0.001 |
| SLIP objective | CLIP - SimCLR | cRSA | -0.05 [-0.06, -0.03], p < 0.001 | -0.017 [-0.053, +0.020], p 0.358 |
| SLIP objective | CLIP - SimCLR | wRSA/veRSA | -0.02 [-0.04, -0.01], p = 0.005 | -0.010 [-0.027, +0.006], p 0.223 |
| SLIP objective | SLIP - SimCLR | cRSA | No substantial effect reported | +0.018 [-0.019, +0.055], p 0.324 |
| SLIP objective | SLIP - SimCLR | wRSA/veRSA | No substantial effect reported | +0.003 [-0.013, +0.020], p 0.681 |
| Input diet | ImageNet21K - ImageNet1K | cRSA | 0.00 [-0.03, 0.03], p = 0.957 | +0.006 [-0.003, +0.015], p 0.212 |
| Input diet | ImageNet21K - ImageNet1K | wRSA/veRSA | +0.01 [0.00, 0.03], p = 0.147 | -0.008 [-0.017, +0.001], p 0.099 |
| IPCL input | OpenImages - ImageNet | cRSA | -0.02 [-0.03, -0.01], p = 0.002 | -0.024 [-0.045, -0.004], p 0.024 |
| IPCL input | OpenImages - ImageNet | wRSA/veRSA | -0.04 [-0.07, -0.02], p < 0.001 | -0.027 [-0.096, +0.042], p 0.406 |
| IPCL input | Places365 - ImageNet | cRSA | -0.03 [-0.04, -0.02], p < 0.001 | -0.077 [-0.097, -0.056], p <0.001 |
| IPCL input | Places365 - ImageNet | wRSA/veRSA | -0.07 [-0.09, -0.04], p < 0.001 | -0.102 [-0.171, -0.033], p 0.007 |
| IPCL input | VGGFace2 - ImageNet | cRSA | -0.17 [-0.18, -0.16], p < 0.001 | -0.003 [-0.024, +0.017], p 0.722 |
| IPCL input | VGGFace2 - ImageNet | wRSA/veRSA | -0.27 [-0.30, -0.25], p < 0.001 | -0.100 [-0.169, -0.030], p 0.009 |

#### Taskonomy Endpoints

| task | metric | Conwell paper | split-half |
| --- | --- | --- | --- |
| autoencoding | cRSA | 0.077 [0.066, 0.085] | 0.040 |
| autoencoding | wRSA/veRSA | 0.103 [0.096, 0.110] | -0.080 |
| class_object | cRSA | 0.189 [0.178, 0.201] | 0.087 |
| class_object | wRSA/veRSA | 0.436 [0.419, 0.454] | 0.239 |

### Odd/Even Split-Half + OOD

#### Score Summary

| variant | metric | n | subjects | models | mean | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split-half + OOD | cRSA | 585 | 5 | 117 | 0.239 | 0.253 | -0.229 | 0.547 |
| split-half + OOD | wRSA/veRSA | 585 | 5 | 117 | 0.506 | 0.527 | -0.028 | 0.648 |
| split-half + OOD | SRPR | 585 | 5 | 117 | 0.387 | 0.422 | 0.091 | 0.475 |

#### Controlled-Effect Comparison

| family | effect | metric | Conwell paper | split-half + OOD |
| --- | --- | --- | --- | --- |
| Architecture | Transformer - CNN | cRSA | -0.04 [-0.05, -0.03], p < 0.001 | -0.087 [-0.115, -0.060], p <0.001 |
| Architecture | Transformer - CNN | wRSA/veRSA | -0.01 [-0.02, -0.00], p < 0.001 | +0.047 [+0.036, +0.057], p <0.001 |
| Self-supervised | Contrastive - Non-Contrastive | cRSA | +0.06 contrastive advantage, p < 0.001 | +0.099 [+0.074, +0.124], p <0.001 |
| Self-supervised | Contrastive - Non-Contrastive | wRSA/veRSA | +0.09 contrastive advantage, p < 0.001 | +0.006 [-0.018, +0.030], p 0.612 |
| SLIP objective | CLIP - SimCLR | cRSA | -0.05 [-0.06, -0.03], p < 0.001 | -0.172 [-0.271, -0.072], p 0.001 |
| SLIP objective | CLIP - SimCLR | wRSA/veRSA | -0.02 [-0.04, -0.01], p = 0.005 | +0.012 [-0.000, +0.024], p 0.056 |
| SLIP objective | SLIP - SimCLR | cRSA | No substantial effect reported | +0.028 [-0.071, +0.128], p 0.566 |
| SLIP objective | SLIP - SimCLR | wRSA/veRSA | No substantial effect reported | +0.004 [-0.008, +0.017], p 0.468 |
| Input diet | ImageNet21K - ImageNet1K | cRSA | 0.00 [-0.03, 0.03], p = 0.957 | +0.013 [-0.002, +0.027], p 0.090 |
| Input diet | ImageNet21K - ImageNet1K | wRSA/veRSA | +0.01 [0.00, 0.03], p = 0.147 | +0.001 [-0.007, +0.009], p 0.774 |
| IPCL input | OpenImages - ImageNet | cRSA | -0.02 [-0.03, -0.01], p = 0.002 | -0.004 [-0.040, +0.032], p 0.817 |
| IPCL input | OpenImages - ImageNet | wRSA/veRSA | -0.04 [-0.07, -0.02], p < 0.001 | -0.034 [-0.064, -0.004], p 0.028 |
| IPCL input | Places365 - ImageNet | cRSA | -0.03 [-0.04, -0.02], p < 0.001 | -0.052 [-0.088, -0.016], p 0.009 |
| IPCL input | Places365 - ImageNet | wRSA/veRSA | -0.07 [-0.09, -0.04], p < 0.001 | -0.057 [-0.087, -0.027], p 0.001 |
| IPCL input | VGGFace2 - ImageNet | cRSA | -0.17 [-0.18, -0.16], p < 0.001 | +0.116 [+0.080, +0.152], p <0.001 |
| IPCL input | VGGFace2 - ImageNet | wRSA/veRSA | -0.27 [-0.30, -0.25], p < 0.001 | -0.057 [-0.087, -0.028], p 0.001 |

#### Taskonomy Endpoints

| task | metric | Conwell paper | split-half + OOD |
| --- | --- | --- | --- |
| autoencoding | cRSA | 0.077 [0.066, 0.085] | 0.109 |
| autoencoding | wRSA/veRSA | 0.103 [0.096, 0.110] | 0.064 |
| class_object | cRSA | 0.189 [0.178, 0.201] | 0.175 |
| class_object | wRSA/veRSA | 0.436 [0.419, 0.454] | 0.492 |

### Random Splits, Averaged Across 5 Repeats

#### Score Summary

| variant | metric | n | subjects | models | mean | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared-image pool | cRSA | 585 | 5 | 117 | 0.192 | 0.186 | 0.010 | 0.412 |
| shared-image pool | wRSA/veRSA | 585 | 5 | 117 | 0.423 | 0.471 | -0.054 | 0.621 |
| shared-image pool | SRPR | 585 | 5 | 117 | 0.399 | 0.437 | 0.111 | 0.499 |
| subject-image pool | cRSA | 585 | 5 | 117 | 0.174 | 0.173 | 0.014 | 0.346 |
| subject-image pool | wRSA/veRSA | 585 | 5 | 117 | 0.422 | 0.454 | -0.010 | 0.625 |
| subject-image pool | SRPR | 585 | 5 | 117 | 0.366 | 0.399 | 0.133 | 0.449 |

#### Controlled-Effect Comparison

| family | effect | metric | Conwell paper | shared-image pool | subject-image pool |
| --- | --- | --- | --- | --- | --- |
| Architecture | Transformer - CNN | cRSA | -0.04 [-0.05, -0.03], p < 0.001 | -0.080 [-0.094, -0.066], p <0.001 | -0.057 [-0.068, -0.045], p <0.001 |
| Architecture | Transformer - CNN | wRSA/veRSA | -0.01 [-0.02, -0.00], p < 0.001 | +0.018 [+0.009, +0.028], p <0.001 | +0.014 [+0.006, +0.022], p <0.001 |
| Self-supervised | Contrastive - Non-Contrastive | cRSA | +0.06 contrastive advantage, p < 0.001 | +0.110 [+0.090, +0.131], p <0.001 | +0.059 [+0.041, +0.078], p <0.001 |
| Self-supervised | Contrastive - Non-Contrastive | wRSA/veRSA | +0.09 contrastive advantage, p < 0.001 | +0.079 [+0.063, +0.096], p <0.001 | +0.050 [+0.036, +0.065], p <0.001 |
| SLIP objective | CLIP - SimCLR | cRSA | -0.05 [-0.06, -0.03], p < 0.001 | -0.050 [-0.089, -0.010], p 0.016 | -0.051 [-0.078, -0.024], p <0.001 |
| SLIP objective | CLIP - SimCLR | wRSA/veRSA | -0.02 [-0.04, -0.01], p = 0.005 | -0.018 [-0.032, -0.003], p 0.021 | -0.020 [-0.031, -0.010], p <0.001 |
| SLIP objective | SLIP - SimCLR | cRSA | No substantial effect reported | +0.001 [-0.039, +0.041], p 0.957 | +0.012 [-0.016, +0.039], p 0.393 |
| SLIP objective | SLIP - SimCLR | wRSA/veRSA | No substantial effect reported | +0.008 [-0.007, +0.023], p 0.279 | +0.007 [-0.003, +0.018], p 0.164 |
| Input diet | ImageNet21K - ImageNet1K | cRSA | 0.00 [-0.03, 0.03], p = 0.957 | +0.011 [+0.003, +0.020], p 0.006 | +0.012 [+0.004, +0.019], p 0.002 |
| Input diet | ImageNet21K - ImageNet1K | wRSA/veRSA | +0.01 [0.00, 0.03], p = 0.147 | -0.012 [-0.020, -0.004], p 0.004 | -0.009 [-0.015, -0.004], p <0.001 |
| IPCL input | OpenImages - ImageNet | cRSA | -0.02 [-0.03, -0.01], p = 0.002 | -0.037 [-0.052, -0.021], p <0.001 | -0.023 [-0.032, -0.014], p <0.001 |
| IPCL input | OpenImages - ImageNet | wRSA/veRSA | -0.04 [-0.07, -0.02], p < 0.001 | -0.006 [-0.067, +0.055], p 0.837 | -0.018 [-0.046, +0.010], p 0.194 |
| IPCL input | Places365 - ImageNet | cRSA | -0.03 [-0.04, -0.02], p < 0.001 | -0.095 [-0.110, -0.079], p <0.001 | -0.077 [-0.086, -0.068], p <0.001 |
| IPCL input | Places365 - ImageNet | wRSA/veRSA | -0.07 [-0.09, -0.04], p < 0.001 | -0.047 [-0.108, +0.014], p 0.120 | -0.059 [-0.087, -0.031], p <0.001 |
| IPCL input | VGGFace2 - ImageNet | cRSA | -0.17 [-0.18, -0.16], p < 0.001 | +0.016 [+0.001, +0.032], p 0.043 | +0.001 [-0.008, +0.010], p 0.832 |
| IPCL input | VGGFace2 - ImageNet | wRSA/veRSA | -0.27 [-0.30, -0.25], p < 0.001 | -0.058 [-0.119, +0.003], p 0.062 | -0.119 [-0.147, -0.090], p <0.001 |

#### Taskonomy Endpoints

| task | metric | Conwell paper | shared-image pool | subject-image pool |
| --- | --- | --- | --- | --- |
| autoencoding | cRSA | 0.077 [0.066, 0.085] | 0.053 | 0.034 |
| autoencoding | wRSA/veRSA | 0.103 [0.096, 0.110] | -0.042 | -0.002 |
| class_object | cRSA | 0.189 [0.178, 0.201] | 0.118 | 0.117 |
| class_object | wRSA/veRSA | 0.436 [0.419, 0.454] | 0.325 | 0.316 |

### Cluster Splits, Averaged Across 5 Repeats

#### Score Summary

| variant | metric | n | subjects | models | mean | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared-image pool | cRSA | 585 | 5 | 117 | 0.171 | 0.167 | -0.036 | 0.416 |
| shared-image pool | wRSA/veRSA | 585 | 5 | 117 | 0.318 | 0.338 | -0.074 | 0.498 |
| shared-image pool | SRPR | 585 | 5 | 117 | 0.304 | 0.337 | 0.072 | 0.392 |
| subject-image pool | cRSA | 585 | 5 | 117 | 0.169 | 0.168 | 0.003 | 0.342 |
| subject-image pool | wRSA/veRSA | 585 | 5 | 117 | 0.344 | 0.356 | -0.018 | 0.522 |
| subject-image pool | SRPR | 585 | 5 | 117 | 0.295 | 0.325 | 0.082 | 0.376 |

#### Controlled-Effect Comparison

| family | effect | metric | Conwell paper | shared-image pool | subject-image pool |
| --- | --- | --- | --- | --- | --- |
| Architecture | Transformer - CNN | cRSA | -0.04 [-0.05, -0.03], p < 0.001 | -0.051 [-0.064, -0.037], p <0.001 | -0.043 [-0.054, -0.032], p <0.001 |
| Architecture | Transformer - CNN | wRSA/veRSA | -0.01 [-0.02, -0.00], p < 0.001 | +0.032 [+0.022, +0.042], p <0.001 | +0.021 [+0.011, +0.030], p <0.001 |
| Self-supervised | Contrastive - Non-Contrastive | cRSA | +0.06 contrastive advantage, p < 0.001 | +0.087 [+0.064, +0.110], p <0.001 | +0.076 [+0.059, +0.093], p <0.001 |
| Self-supervised | Contrastive - Non-Contrastive | wRSA/veRSA | +0.09 contrastive advantage, p < 0.001 | +0.086 [+0.069, +0.104], p <0.001 | +0.056 [+0.042, +0.070], p <0.001 |
| SLIP objective | CLIP - SimCLR | cRSA | -0.05 [-0.06, -0.03], p < 0.001 | -0.077 [-0.111, -0.043], p <0.001 | -0.083 [-0.112, -0.055], p <0.001 |
| SLIP objective | CLIP - SimCLR | wRSA/veRSA | -0.02 [-0.04, -0.01], p = 0.005 | -0.040 [-0.056, -0.024], p <0.001 | -0.022 [-0.035, -0.008], p 0.003 |
| SLIP objective | SLIP - SimCLR | cRSA | No substantial effect reported | +0.005 [-0.029, +0.040], p 0.748 | -0.004 [-0.033, +0.025], p 0.795 |
| SLIP objective | SLIP - SimCLR | wRSA/veRSA | No substantial effect reported | -0.021 [-0.036, -0.005], p 0.013 | -0.001 [-0.014, +0.013], p 0.927 |
| Input diet | ImageNet21K - ImageNet1K | cRSA | 0.00 [-0.03, 0.03], p = 0.957 | +0.003 [-0.004, +0.010], p 0.452 | -0.000 [-0.008, +0.008], p 0.999 |
| Input diet | ImageNet21K - ImageNet1K | wRSA/veRSA | +0.01 [0.00, 0.03], p = 0.147 | -0.011 [-0.021, -0.000], p 0.045 | -0.016 [-0.025, -0.007], p <0.001 |
| IPCL input | OpenImages - ImageNet | cRSA | -0.02 [-0.03, -0.01], p = 0.002 | -0.013 [-0.037, +0.012], p 0.277 | -0.021 [-0.035, -0.008], p 0.005 |
| IPCL input | OpenImages - ImageNet | wRSA/veRSA | -0.04 [-0.07, -0.02], p < 0.001 | +0.000 [-0.040, +0.041], p 0.989 | -0.022 [-0.038, -0.005], p 0.014 |
| IPCL input | Places365 - ImageNet | cRSA | -0.03 [-0.04, -0.02], p < 0.001 | -0.057 [-0.082, -0.033], p <0.001 | -0.062 [-0.076, -0.048], p <0.001 |
| IPCL input | Places365 - ImageNet | wRSA/veRSA | -0.07 [-0.09, -0.04], p < 0.001 | -0.050 [-0.091, -0.010], p 0.019 | -0.053 [-0.069, -0.036], p <0.001 |
| IPCL input | VGGFace2 - ImageNet | cRSA | -0.17 [-0.18, -0.16], p < 0.001 | +0.023 [-0.001, +0.047], p 0.060 | +0.007 [-0.007, +0.020], p 0.310 |
| IPCL input | VGGFace2 - ImageNet | wRSA/veRSA | -0.27 [-0.30, -0.25], p < 0.001 | -0.004 [-0.045, +0.036], p 0.825 | -0.060 [-0.076, -0.043], p <0.001 |

#### Taskonomy Endpoints

| task | metric | Conwell paper | shared-image pool | subject-image pool |
| --- | --- | --- | --- | --- |
| autoencoding | cRSA | 0.077 [0.066, 0.085] | 0.057 | 0.042 |
| autoencoding | wRSA/veRSA | 0.103 [0.096, 0.110] | -0.043 | -0.005 |
| class_object | cRSA | 0.189 [0.178, 0.201] | 0.142 | 0.136 |
| class_object | wRSA/veRSA | 0.436 [0.419, 0.454] | 0.266 | 0.259 |

### Min-NN Threshold Split

#### Score Summary

| variant | metric | n | subjects | models | mean | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared-image pool | cRSA | 585 | 5 | 117 | 0.172 | 0.174 | -0.078 | 0.388 |
| shared-image pool | wRSA/veRSA | 585 | 5 | 117 | 0.386 | 0.405 | -0.050 | 0.645 |
| shared-image pool | SRPR | 585 | 5 | 117 | 0.384 | 0.421 | 0.097 | 0.488 |
| subject-image pool | cRSA | 585 | 5 | 117 | 0.171 | 0.169 | 0.005 | 0.330 |
| subject-image pool | wRSA/veRSA | 585 | 5 | 117 | 0.415 | 0.447 | -0.021 | 0.632 |
| subject-image pool | SRPR | 585 | 5 | 117 | 0.367 | 0.400 | 0.123 | 0.446 |

#### Controlled-Effect Comparison

| family | effect | metric | Conwell paper | shared-image pool | subject-image pool |
| --- | --- | --- | --- | --- | --- |
| Architecture | Transformer - CNN | cRSA | -0.04 [-0.05, -0.03], p < 0.001 | -0.059 [-0.077, -0.042], p <0.001 | -0.060 [-0.072, -0.048], p <0.001 |
| Architecture | Transformer - CNN | wRSA/veRSA | -0.01 [-0.02, -0.00], p < 0.001 | +0.036 [+0.023, +0.049], p <0.001 | +0.016 [+0.007, +0.024], p <0.001 |
| Self-supervised | Contrastive - Non-Contrastive | cRSA | +0.06 contrastive advantage, p < 0.001 | +0.131 [+0.107, +0.155], p <0.001 | +0.065 [+0.046, +0.084], p <0.001 |
| Self-supervised | Contrastive - Non-Contrastive | wRSA/veRSA | +0.09 contrastive advantage, p < 0.001 | +0.080 [+0.052, +0.107], p <0.001 | +0.056 [+0.040, +0.071], p <0.001 |
| SLIP objective | CLIP - SimCLR | cRSA | -0.05 [-0.06, -0.03], p < 0.001 | -0.031 [-0.083, +0.021], p 0.237 | -0.029 [-0.057, -0.000], p 0.049 |
| SLIP objective | CLIP - SimCLR | wRSA/veRSA | -0.02 [-0.04, -0.01], p = 0.005 | +0.009 [-0.020, +0.037], p 0.534 | -0.024 [-0.037, -0.011], p <0.001 |
| SLIP objective | SLIP - SimCLR | cRSA | No substantial effect reported | +0.034 [-0.018, +0.087], p 0.190 | +0.014 [-0.014, +0.043], p 0.314 |
| SLIP objective | SLIP - SimCLR | wRSA/veRSA | No substantial effect reported | +0.010 [-0.018, +0.039], p 0.462 | +0.007 [-0.006, +0.020], p 0.280 |
| Input diet | ImageNet21K - ImageNet1K | cRSA | 0.00 [-0.03, 0.03], p = 0.957 | -0.005 [-0.021, +0.010], p 0.498 | +0.012 [+0.003, +0.021], p 0.013 |
| Input diet | ImageNet21K - ImageNet1K | wRSA/veRSA | +0.01 [0.00, 0.03], p = 0.147 | -0.031 [-0.044, -0.018], p <0.001 | -0.009 [-0.015, -0.002], p 0.014 |
| IPCL input | OpenImages - ImageNet | cRSA | -0.02 [-0.03, -0.01], p = 0.002 | -0.056 [-0.085, -0.028], p <0.001 | -0.014 [-0.030, +0.002], p 0.088 |
| IPCL input | OpenImages - ImageNet | wRSA/veRSA | -0.04 [-0.07, -0.02], p < 0.001 | -0.042 [-0.165, +0.080], p 0.468 | -0.016 [-0.053, +0.021], p 0.377 |
| IPCL input | Places365 - ImageNet | cRSA | -0.03 [-0.04, -0.02], p < 0.001 | -0.111 [-0.139, -0.083], p <0.001 | -0.075 [-0.092, -0.059], p <0.001 |
| IPCL input | Places365 - ImageNet | wRSA/veRSA | -0.07 [-0.09, -0.04], p < 0.001 | -0.081 [-0.203, +0.042], p 0.177 | -0.044 [-0.081, -0.007], p 0.022 |
| IPCL input | VGGFace2 - ImageNet | cRSA | -0.17 [-0.18, -0.16], p < 0.001 | +0.003 [-0.025, +0.031], p 0.828 | -0.011 [-0.027, +0.006], p 0.184 |
| IPCL input | VGGFace2 - ImageNet | wRSA/veRSA | -0.27 [-0.30, -0.25], p < 0.001 | -0.029 [-0.151, +0.093], p 0.616 | -0.115 [-0.152, -0.078], p <0.001 |

#### Taskonomy Endpoints

| task | metric | Conwell paper | shared-image pool | subject-image pool |
| --- | --- | --- | --- | --- |
| autoencoding | cRSA | 0.077 [0.066, 0.085] | 0.059 | 0.040 |
| autoencoding | wRSA/veRSA | 0.103 [0.096, 0.110] | -0.020 | 0.007 |
| class_object | cRSA | 0.189 [0.178, 0.201] | 0.121 | 0.115 |
| class_object | wRSA/veRSA | 0.436 [0.419, 0.454] | 0.250 | 0.300 |

### OOD Test Split

#### Score Summary

| variant | metric | n | subjects | models | mean | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared-image pool | cRSA | 585 | 5 | 117 | 0.081 | 0.077 | -0.329 | 0.639 |
| shared-image pool | wRSA/veRSA | 585 | 5 | 117 | 0.217 | 0.226 | -0.040 | 0.604 |
| shared-image pool | SRPR | 585 | 5 | 117 | 0.319 | 0.347 | -0.037 | 0.415 |
| subject-image pool | cRSA | 585 | 5 | 117 | 0.073 | 0.063 | -0.329 | 0.639 |
| subject-image pool | wRSA/veRSA | 585 | 5 | 117 | 0.230 | 0.231 | -0.018 | 0.620 |
| subject-image pool | SRPR | 585 | 5 | 117 | 0.343 | 0.366 | -0.058 | 0.433 |

#### Controlled-Effect Comparison

| family | effect | metric | Conwell paper | shared-image pool | subject-image pool |
| --- | --- | --- | --- | --- | --- |
| Architecture | Transformer - CNN | cRSA | -0.04 [-0.05, -0.03], p < 0.001 | -0.065 [-0.093, -0.038], p <0.001 | -0.071 [-0.099, -0.044], p <0.001 |
| Architecture | Transformer - CNN | wRSA/veRSA | -0.01 [-0.02, -0.00], p < 0.001 | +0.022 [+0.008, +0.036], p 0.002 | +0.030 [+0.015, +0.045], p <0.001 |
| Self-supervised | Contrastive - Non-Contrastive | cRSA | +0.06 contrastive advantage, p < 0.001 | +0.161 [+0.112, +0.210], p <0.001 | +0.159 [+0.116, +0.202], p <0.001 |
| Self-supervised | Contrastive - Non-Contrastive | wRSA/veRSA | +0.09 contrastive advantage, p < 0.001 | +0.070 [+0.009, +0.131], p 0.025 | +0.051 [-0.009, +0.110], p 0.093 |
| SLIP objective | CLIP - SimCLR | cRSA | -0.05 [-0.06, -0.03], p < 0.001 | -0.138 [-0.244, -0.032], p 0.012 | -0.157 [-0.255, -0.059], p 0.002 |
| SLIP objective | CLIP - SimCLR | wRSA/veRSA | -0.02 [-0.04, -0.01], p = 0.005 | +0.038 [+0.006, +0.069], p 0.019 | +0.035 [+0.006, +0.065], p 0.021 |
| SLIP objective | SLIP - SimCLR | cRSA | No substantial effect reported | +0.033 [-0.073, +0.139], p 0.530 | -0.000 [-0.098, +0.098], p 0.997 |
| SLIP objective | SLIP - SimCLR | wRSA/veRSA | No substantial effect reported | +0.007 [-0.024, +0.038], p 0.640 | -0.020 [-0.050, +0.010], p 0.178 |
| Input diet | ImageNet21K - ImageNet1K | cRSA | 0.00 [-0.03, 0.03], p = 0.957 | -0.018 [-0.044, +0.008], p 0.169 | -0.041 [-0.067, -0.015], p 0.002 |
| Input diet | ImageNet21K - ImageNet1K | wRSA/veRSA | +0.01 [0.00, 0.03], p = 0.147 | -0.018 [-0.034, -0.002], p 0.029 | -0.029 [-0.047, -0.010], p 0.003 |
| IPCL input | OpenImages - ImageNet | cRSA | -0.02 [-0.03, -0.01], p = 0.002 | -0.015 [-0.108, +0.077], p 0.722 | -0.015 [-0.106, +0.075], p 0.718 |
| IPCL input | OpenImages - ImageNet | wRSA/veRSA | -0.04 [-0.07, -0.02], p < 0.001 | +0.021 [-0.056, +0.098], p 0.557 | +0.055 [-0.029, +0.140], p 0.180 |
| IPCL input | Places365 - ImageNet | cRSA | -0.03 [-0.04, -0.02], p < 0.001 | -0.045 [-0.137, +0.047], p 0.310 | -0.049 [-0.139, +0.042], p 0.266 |
| IPCL input | Places365 - ImageNet | wRSA/veRSA | -0.07 [-0.09, -0.04], p < 0.001 | +0.012 [-0.065, +0.089], p 0.739 | +0.077 [-0.008, +0.162], p 0.071 |
| IPCL input | VGGFace2 - ImageNet | cRSA | -0.17 [-0.18, -0.16], p < 0.001 | +0.207 [+0.115, +0.299], p <0.001 | +0.211 [+0.121, +0.302], p <0.001 |
| IPCL input | VGGFace2 - ImageNet | wRSA/veRSA | -0.27 [-0.30, -0.25], p < 0.001 | +0.196 [+0.119, +0.273], p <0.001 | +0.198 [+0.113, +0.282], p <0.001 |

#### Taskonomy Endpoints

| task | metric | Conwell paper | shared-image pool | subject-image pool |
| --- | --- | --- | --- | --- |
| autoencoding | cRSA | 0.077 [0.066, 0.085] | 0.016 | 0.016 |
| autoencoding | wRSA/veRSA | 0.103 [0.096, 0.110] | 0.076 | 0.093 |
| class_object | cRSA | 0.189 [0.178, 0.201] | 0.032 | 0.008 |
| class_object | wRSA/veRSA | 0.436 [0.419, 0.454] | 0.236 | 0.181 |
