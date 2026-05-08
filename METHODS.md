# Methods

## Overview

We evaluated candidate visual neural network models on the LAION-fMRI dataset using a pipeline designed to closely follow the feature-extraction and representational-similarity mapping procedures of Conwell et al. (2024), while replacing the NSD stimulus set and brain measurements with the LAION-fMRI benchmark. The pipeline has five main stages: construction of image pools and train/test splits, voxel and ROI selection, DeepNSD-style model feature extraction, representational-similarity evaluation, and brain-only noise-ceiling estimation.

All analyses were performed at the single-subject level and then summarized across subjects. The evaluation subjects were the five LAION-fMRI participants available in the replication pipeline: `sub-01`, `sub-03`, `sub-05`, `sub-06`, and `sub-07`.

## Stimulus Pools and Evaluation Splits

We evaluated models on two classes of LAION-fMRI stimulus pools. First, for the split-half replication analysis, we used the cross-subject shared stimulus pool. This analysis follows the logic of the Conwell et al. (2024) train/test procedure: images were split into alternating even and odd positions in the shared-stimulus ordering, with one half used for layer selection and model fitting and the other half used for held-out testing. Out-of-distribution shared images with IDs beginning `shared_4rep_OOD_` were excluded from the default split-half analysis.

Second, for generalization analyses, we used the split definitions distributed with `laion_fmri.splits`. These included five random train/test splits, five cluster-holdout splits (`cluster_k5_*`), a distribution-matched `tau` split, and an OOD split. These splits were evaluated either on the cross-subject shared pool or on each subject's full subject-specific pool. For the OOD split, the model was fit on regular in-distribution images and evaluated both on the aggregate OOD test set and, where applicable, on OOD-category subsets.

To avoid repeated feature extraction, we built a deduplicated union image pool across all subject-specific pools. The feature extractor writes image-ID-indexed HDF5 files, allowing the evaluators to align model features to each subject and split by image ID rather than by row position.

## Brain Response Preparation

For each subject, trial-level GLMsingle beta estimates were loaded from the LAION-fMRI derivatives. Repeated presentations of the same image were averaged to obtain one response vector per image. The resulting response matrix had shape voxels by images, and all downstream evaluation code selected train and test responses by image ID.

For efficient repeated evaluation, we cached trial-level beta matrices and aligned image IDs for each subject. The cache was built from voxels passing a broad responsive-cortex threshold based on the mean GLM \(R^2\) map (`R^2 > 0.15`). Voxels containing NaNs in any trial were removed during cache construction.

## Voxel Selection Procedure

As in Conwell et al. (2024), we applied a reliability-based voxel selection procedure before evaluating model-brain correspondence. Because LAION-fMRI does not provide the NSD metadata used directly by Conwell et al., we computed an analogous per-voxel NCSNR metric from the repeated-trial LAION-fMRI betas. Betas were z-scored per voxel, and for each voxel we estimated trial-level noise as the square root of the mean across-stimulus response variance over repeated presentations. Signal standard deviation was then estimated as

\[
\sigma_\mathrm{signal} = \sqrt{\max(0, 1 - \sigma_\mathrm{noise}^2)}
\]

and NCSNR was computed as \(\sigma_\mathrm{signal} / \sigma_\mathrm{noise}\). Only stimuli with at least two repetitions contributed to this estimate. In the main analyses, we retained voxels with `NCSNR > 0.2`, matching the threshold used by Conwell et al. (2024).

After reliability filtering, analyses were restricted to occipitotemporal cortex (OTC), intended as a LAION-fMRI analogue of the human IT/OTC sector used by Conwell et al. (2024). OTC was defined from the available LAION-fMRI ROI masks as the union of a stream mask and category-selective masks. The stream mask was the intersection of `laion-general` with the union of `laion-ventral` and `laion-lateral`. Category-selective masks included OFA, FFA-1, FFA-2, EBA, FBA, VWFA-1, VWFA-2, mfs-words, pSTS-words, PPA, and OPA when available. We also retained component ROI labels so that scores could be reported for OTC as a whole and for its constituent subregions.

In implementation, the OTC mask was applied while building the cached beta matrix, and the NCSNR threshold was applied when constructing each evaluation benchmark. Thus, all reported OTC scores were computed on the intersection of the responsive-cortex mask, the OTC ROI mask, non-NaN voxels, and the `NCSNR > 0.2` reliability mask.

## Model Set and Feature Extraction

Model loading and preprocessing followed Conwell et al. (2024) as closely as possible by using the DeepNSD model registry and model-specific preprocessing definitions. For each model, images were transformed using DeepNSD's recommended test-time image transforms for that model. These transforms typically consisted of resizing/cropping followed by conversion to tensor space and pixel normalization using model-appropriate image statistics.

Features were extracted with DeepNSD's hook-based feature extraction machinery. A layer was defined as a distinct PyTorch module, following the module-level convention of Conwell et al. (2024). Feature maps were flattened after extraction. No brain data entered the feature-extraction stage.

For computational tractability, we selected a subset of layers from each model rather than storing every module output. Specifically, we ignored uninformative container or pass-through module types such as `Dropout`, `Identity`, `Flatten`, `Sequential`, and `ModuleList`, then sampled approximately ten layers from the second half of the remaining model depth, always retaining the final layer. Layer names were stored in the same `ModuleType-N` style used by DeepNSD.

For each selected layer, flattened features were reduced with sparse random projection, as in Conwell et al. (2024). We used the scikit-learn sparse random projection implementation with a fixed random seed and a fixed target dimensionality of 5,960 projected features per layer. This target dimensionality follows the Johnson-Lindenstrauss sparse-projection logic used by Conwell et al. for preserving representational geometry while making encoding fits computationally feasible. Reduced features were written to HDF5 files under `/features_srp/<layer>`, together with a row-aligned `image_ids` dataset.

## Classical RSA

Classical representational similarity analysis (cRSA) was computed using the same train-layer-selection and held-out-test logic as Conwell et al. (2024). For each subject, split, model, and layer, we formed a model representational dissimilarity matrix (RDM) from the layer's reduced feature matrix. RDM entries were computed as one minus the Pearson correlation between image feature vectors.

Brain RDMs were computed analogously from the multivoxel beta patterns in the target ROI. A layer's cRSA score was the Pearson correlation between the upper triangles of the model and brain RDMs. For each model and subject, the best layer was selected using the training image set. The final cRSA score was the score of that same layer on the held-out test image set.

## Voxel-Encoding RSA and SRPR

Voxel-encoding RSA followed the logic of Conwell et al. (2024), with ridge regression used as an intermediate feature-to-voxel encoding model. For each model layer, sparse-random-projection features were standardized using statistics from the training images. We then fit a multi-output ridge regression from model features to voxel responses across the training images. Ridge penalties were selected per voxel using generalized leave-one-out cross-validation over the alpha grid `[0.1, 1, 10, 100, 1000, 10000, 100000]`, with Pearson correlation as the alpha-selection score.

The ridge model yielded cross-validated predicted voxel responses for training images and held-out predicted responses for test images. For voxel-encoding RSA (called `wrsa` in the code outputs), predicted multivoxel response patterns were converted into predicted brain RDMs. These predicted RDMs were compared to the observed brain RDMs using Pearson correlation over the upper triangle. As with cRSA, model-layer selection was performed on training scores, and final performance was the held-out test score for the selected layer.

In addition to the population-level RSA score, we computed a single-voxel prediction score (`srpr`). This score was the Pearson correlation between predicted and observed responses for each voxel, averaged across voxels in the target ROI. Voxelwise SRPR vectors were optionally written as sidecar files for later analyses. The main model-comparison analyses prioritized the representational population score, following the motivation of Conwell et al. (2024), but SRPR was retained as a complementary encoding metric.

## Layer Selection and Score Aggregation

For every metric, model, subject, and evaluation split, best-layer selection was performed using training-set scores only. The selected layer was then evaluated on the corresponding held-out test set. For min-nn analyses, split and pool filters were applied before plotting or aggregating scores, so that best-layer selection was performed within the intended evaluation condition. Scores were summarized across subjects after layer selection.

Controlled comparison figures used the Conwell-style model contrast metadata in `resources/model_contrasts.csv`. The evaluation manifest included models participating in the architecture, task, self-supervised, SLIP, ImageNet-scale, and IPCL comparisons for which complete feature files were available.

## Noise Ceilings

Noise ceilings were computed from brain data alone and did not use model features. Unlike Conwell et al. (2024), who estimated within-subject RSM noise ceilings using their GSN procedure, our primary implemented noise ceiling uses repeated-trial split-half reliability in LAION-fMRI. This choice provides a direct reliability estimate for the target dataset and ROI while preserving the same interpretive role: the ceiling indicates the expected maximum model-brain correspondence given measurement noise in the subject's responses.

For each subject and ROI, trial repetitions for each stimulus were randomly split into two halves. Responses were averaged within each half, producing two independent multivoxel response estimates per stimulus. We then constructed two brain RDMs and correlated their upper triangles using Pearson correlation. This split-half reliability was averaged across repeated random half-splits and Spearman-Brown corrected with \(k = 2\), yielding an ROI-level RSA noise ceiling. The split-half noise ceiling used for the main OTC plots was computed on the same shared, non-OOD stimulus pool and the same even/odd train/test stimulus partition as the split-half evaluator, after applying the same NCSNR voxel filter.

For SRPR, we computed a separate per-voxel reliability estimate using the same z-scored repeated-trial beta formulation used for NCSNR. Per-voxel noise-ceiling percentages were averaged across voxels within the ROI and converted to a Pearson-\(r\)-equivalent value by taking \(\sqrt{\mathrm{NC}/100}\).

For generalization-split plots, the same OTC split-half noise ceiling can be used as a common reference line. This reference is useful for contextualizing the absolute scale of scores, but it should not be interpreted as a split-specific ceiling for every min-nn split, subject-specific pool, or OOD category unless recomputed on the corresponding image set.

## Software and Outputs

Feature extraction, evaluation, and plotting were implemented in the `conwell_replication` package. Feature files were stored as one HDF5 file per model with image-ID-indexed rows. Evaluation outputs were written as parquet tables containing one row per subject, model, layer, metric, ROI, split, and score set. These tables were then filtered to the desired ROI and split condition, layer-selected using training scores, and plotted as controlled comparison figures following the layout and interpretation of Conwell et al. (2024).
