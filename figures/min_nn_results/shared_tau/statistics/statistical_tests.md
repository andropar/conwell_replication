# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.059291 | -0.0766688 | -0.0419132 | t | -6.71795 | 1.12828e-10 | ols_fixed_effects | 270 |
| architecture | wrsa | Transformer | 0.0357447 | 0.0227807 | 0.0487088 | t | 5.42893 | 1.28497e-07 | ols_fixed_effects | 270 |
| taskonomy_tasks | crsa | autoencoding | -0.0563723 | -0.0961097 | -0.0166349 | t | -2.81594 | 0.00590291 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_object | 0.00562582 | -0.0341116 | 0.0453632 | t | 0.281024 | 0.779296 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_scene | 0.00165364 | -0.0380837 | 0.041391 | t | 0.0826035 | 0.934339 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | curvature | -0.0710087 | -0.110746 | -0.0312713 | t | -3.54707 | 0.000604507 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_euclidean | -0.00272125 | -0.0424586 | 0.0370161 | t | -0.135934 | 0.892158 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_zbuffer | 0.0389004 | -0.000836957 | 0.0786378 | t | 1.94318 | 0.0549241 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_occlusion | -0.103045 | -0.142782 | -0.0633073 | t | -5.14735 | 1.40167e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_texture | 0.131333 | 0.0915961 | 0.171071 | t | 6.56044 | 2.71218e-09 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | egomotion | -0.0646025 | -0.10434 | -0.0248651 | t | -3.22706 | 0.00171134 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | fixated_pose | -0.0578115 | -0.0975489 | -0.0180741 | t | -2.88783 | 0.00479337 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | inpainting | -0.04035 | -0.0800874 | -0.000612639 | t | -2.01559 | 0.0466385 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | jigsaw | 0.0387396 | -0.000997797 | 0.078477 | t | 1.93514 | 0.055916 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints2d | -0.00512174 | -0.0448591 | 0.0346156 | t | -0.255844 | 0.798619 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints3d | 0.0514653 | 0.0117279 | 0.0912027 | t | 2.57082 | 0.0116817 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | nonfixated_pose | -0.0601605 | -0.0998978 | -0.0204231 | t | -3.00517 | 0.00338637 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | normal | 0.00878914 | -0.0309482 | 0.0485265 | t | 0.43904 | 0.661619 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | point_matching | 0.0152053 | -0.0245321 | 0.0549427 | t | 0.759543 | 0.449389 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | random_weights | -0.0776579 | -0.117395 | -0.0379205 | t | -3.87921 | 0.000192033 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | reshading | 0.0557381 | 0.0160007 | 0.0954755 | t | 2.78426 | 0.00646274 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | room_layout | -0.0690992 | -0.108837 | -0.0293618 | t | -3.45168 | 0.000829948 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_semantic | 0.0391416 | -0.000595791 | 0.078879 | t | 1.95522 | 0.0534648 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup25d | -0.00282088 | -0.0425583 | 0.0369165 | t | -0.14091 | 0.888236 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.0847985 | 0.0450611 | 0.124536 | t | 4.2359 | 5.22334e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | vanishing_point | -0.112255 | -0.151992 | -0.0725174 | t | -5.60742 | 1.97139e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | autoencoding | -0.208392 | -0.276553 | -0.140231 | t | -6.06878 | 2.55937e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_object | 0.0617017 | -0.00645928 | 0.129863 | t | 1.79688 | 0.0754998 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_scene | 0.101755 | 0.0335942 | 0.169916 | t | 2.96331 | 0.00383737 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | curvature | -0.214918 | -0.283079 | -0.146757 | t | -6.25885 | 1.08302e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_euclidean | 0.0723373 | 0.00417631 | 0.140498 | t | 2.10661 | 0.0377595 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_zbuffer | 0.0338309 | -0.0343301 | 0.101992 | t | 0.985223 | 0.326991 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_occlusion | -0.00734665 | -0.0755077 | 0.0608143 | t | -0.213949 | 0.83104 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_texture | 0.0305261 | -0.0376348 | 0.0986871 | t | 0.888982 | 0.376235 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | egomotion | -0.0458258 | -0.113987 | 0.0223352 | t | -1.33454 | 0.185184 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | fixated_pose | 0.0150143 | -0.0531467 | 0.0831753 | t | 0.437247 | 0.662914 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | inpainting | -0.0259942 | -0.0941552 | 0.0421668 | t | -0.757002 | 0.450903 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | jigsaw | -0.0606399 | -0.128801 | 0.00752105 | t | -1.76596 | 0.0805823 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints2d | -0.105033 | -0.173194 | -0.0368721 | t | -3.05877 | 0.00288031 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints3d | 0.113683 | 0.0455217 | 0.181844 | t | 3.31067 | 0.0013122 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | nonfixated_pose | -0.0264816 | -0.0946426 | 0.0416794 | t | -0.771197 | 0.442484 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | normal | 0.0180825 | -0.0500785 | 0.0862435 | t | 0.526598 | 0.599687 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | point_matching | 0.0549464 | -0.0132146 | 0.123107 | t | 1.60015 | 0.112851 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | random_weights | -0.155551 | -0.223712 | -0.0873904 | t | -4.52997 | 1.6972e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | reshading | 0.0956669 | 0.0275059 | 0.163828 | t | 2.78601 | 0.00643056 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | room_layout | 0.0303393 | -0.0378217 | 0.0985003 | t | 0.883541 | 0.379151 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_semantic | 0.0575977 | -0.0105633 | 0.125759 | t | 1.67736 | 0.0967253 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup25d | 0.0141656 | -0.0539954 | 0.0823266 | t | 0.412531 | 0.68087 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.133152 | 0.0649914 | 0.201313 | t | 3.87766 | 0.000193091 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | vanishing_point | -0.152978 | -0.221139 | -0.0848173 | t | -4.45503 | 2.26957e-05 | ols_fixed_effects | 125 |
| contrastive_self_supervised | crsa | Contrastive | 0.131287 | 0.107294 | 0.15528 | t | 11.0279 | 3.00079e-14 | ols_fixed_effects | 50 |
| contrastive_self_supervised | wrsa | Contrastive | 0.0795914 | 0.0516977 | 0.107485 | t | 5.75061 | 7.8382e-07 | ols_fixed_effects | 50 |
| language_alignment | crsa | CLIP | -0.0309471 | -0.0831365 | 0.0212422 | t | -1.20042 | 0.237406 | ols_fixed_effects | 45 |
| language_alignment | crsa | SLIP | 0.0344157 | -0.0177736 | 0.086605 | t | 1.33496 | 0.189831 | ols_fixed_effects | 45 |
| language_alignment | wrsa | CLIP | 0.00877825 | -0.0195029 | 0.0370594 | t | 0.628356 | 0.53353 | ols_fixed_effects | 45 |
| language_alignment | wrsa | SLIP | 0.0103849 | -0.0178963 | 0.038666 | t | 0.743358 | 0.461836 | ols_fixed_effects | 45 |
| imagenet_size | crsa | imagenet21k | -0.00527994 | -0.0206602 | 0.0101004 | t | -0.680839 | 0.497501 | ols_fixed_effects_fallback | 120 |
| imagenet_size | wrsa | imagenet21k | -0.0309203 | -0.0440632 | -0.0177775 | z | -4.61108 | 4.0058e-06 | mixedlm_random_intercept | 120 |
| objects_faces_places | crsa | openimages | -0.0563572 | -0.0845182 | -0.0281962 | t | -4.36035 | 0.000927792 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | places256 | -0.110774 | -0.138935 | -0.0826125 | t | -8.57053 | 1.84466e-06 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | vggface2 | 0.00286267 | -0.0252983 | 0.0310237 | t | 0.221484 | 0.82844 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | openimages | -0.0420924 | -0.164508 | 0.080323 | t | -0.749183 | 0.468179 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | places256 | -0.0805686 | -0.202984 | 0.0418468 | t | -1.434 | 0.177113 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | vggface2 | -0.0289429 | -0.151358 | 0.0934725 | t | -0.51514 | 0.615812 | ols_fixed_effects | 20 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 83 | 0.3810592621564865 | 107 | 0.1887488335371017 | 83 | 24 | 10 | 0.006574129002777988 | wrsa | 117 | vit_large_patch16_224_classification | 0.549121356010437 | pure_python_grid_search_two_breakpoints | not_computed |
