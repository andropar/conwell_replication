# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.0654203 | -0.092717 | -0.0381237 | t | -4.71897 | 3.84828e-06 | ols_fixed_effects | 270 |
| architecture | wrsa | Transformer | 0.0221305 | 0.00804802 | 0.036213 | t | 3.09425 | 0.00218516 | ols_fixed_effects | 270 |
| taskonomy_tasks | crsa | autoencoding | -0.0966111 | -0.135555 | -0.0576671 | t | -4.92429 | 3.51932e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_object | -0.0804297 | -0.119374 | -0.0414858 | t | -4.09952 | 8.66404e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_scene | -0.0848137 | -0.123758 | -0.0458697 | t | -4.32298 | 3.76173e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | curvature | -0.0748846 | -0.113829 | -0.0359407 | t | -3.81689 | 0.000239321 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_euclidean | -0.149571 | -0.188515 | -0.110627 | t | -7.62366 | 1.76263e-11 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_zbuffer | -0.111771 | -0.150715 | -0.0728269 | t | -5.69699 | 1.33352e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_occlusion | -0.10721 | -0.146154 | -0.068266 | t | -5.46452 | 3.65642e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_texture | 0.0618583 | 0.0229144 | 0.100802 | t | 3.15294 | 0.00215741 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | egomotion | -0.0680069 | -0.106951 | -0.029063 | t | -3.46633 | 0.00079081 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | fixated_pose | -0.0721816 | -0.111126 | -0.0332376 | t | -3.67911 | 0.000386215 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | inpainting | 0.1148 | 0.0758562 | 0.153744 | t | 5.85139 | 6.75472e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | jigsaw | 0.000397232 | -0.0385467 | 0.0393412 | t | 0.020247 | 0.983888 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints2d | 0.0159516 | -0.0229924 | 0.0548955 | t | 0.813056 | 0.418198 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints3d | -0.0350714 | -0.0740154 | 0.00387253 | t | -1.7876 | 0.076996 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | nonfixated_pose | -0.0290963 | -0.0680402 | 0.00984769 | t | -1.48304 | 0.141338 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | normal | -0.0351235 | -0.0740675 | 0.00382043 | t | -1.79026 | 0.0765652 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | point_matching | 0.071843 | 0.032899 | 0.110787 | t | 3.66186 | 0.000409753 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | random_weights | 0.243534 | 0.20459 | 0.282478 | t | 12.413 | 1.13007e-21 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | reshading | -0.0882621 | -0.127206 | -0.0493181 | t | -4.49874 | 1.91635e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | room_layout | -0.211096 | -0.25004 | -0.172152 | t | -10.7596 | 3.50619e-18 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_semantic | -0.0569153 | -0.0958592 | -0.0179713 | t | -2.90099 | 0.0046124 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup25d | -0.0832361 | -0.12218 | -0.0442922 | t | -4.24257 | 5.09446e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.0292501 | -0.00969387 | 0.068194 | t | 1.49088 | 0.13927 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | vanishing_point | -0.201014 | -0.239958 | -0.16207 | t | -10.2457 | 4.41477e-17 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | autoencoding | -0.214381 | -0.289489 | -0.139272 | t | -5.66569 | 1.5292e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_object | -0.0542524 | -0.129361 | 0.0208562 | t | -1.43379 | 0.154881 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_scene | -0.0708864 | -0.145995 | 0.00422228 | t | -1.8734 | 0.0640557 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | curvature | -0.313931 | -0.389039 | -0.238822 | t | -8.29661 | 6.66987e-13 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_euclidean | -0.145952 | -0.221061 | -0.0708435 | t | -3.85725 | 0.000207577 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_zbuffer | -0.0969559 | -0.172065 | -0.0218472 | t | -2.56237 | 0.011951 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_occlusion | -0.0731944 | -0.148303 | 0.00191429 | t | -1.93439 | 0.0560092 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_texture | -0.051241 | -0.12635 | 0.0238677 | t | -1.35421 | 0.17885 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | egomotion | -0.0690892 | -0.144198 | 0.00601949 | t | -1.8259 | 0.0709739 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | fixated_pose | -0.214839 | -0.289948 | -0.13973 | t | -5.6778 | 1.45035e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | inpainting | 0.162587 | 0.0874788 | 0.237696 | t | 4.29689 | 4.15223e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | jigsaw | -0.273186 | -0.348295 | -0.198078 | t | -7.21981 | 1.22293e-10 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints2d | -0.17926 | -0.254369 | -0.104152 | t | -4.73752 | 7.48142e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints3d | 0.147417 | 0.0723078 | 0.222525 | t | 3.89595 | 0.00018094 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | nonfixated_pose | -0.294403 | -0.369511 | -0.219294 | t | -7.78052 | 8.25409e-12 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | normal | -0.0427562 | -0.117865 | 0.0323525 | t | -1.12997 | 0.261305 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | point_matching | -0.139474 | -0.214583 | -0.0643655 | t | -3.68605 | 0.00037713 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | random_weights | -0.048679 | -0.123788 | 0.0264297 | t | -1.2865 | 0.201364 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | reshading | -0.0708379 | -0.145947 | 0.0042708 | t | -1.87212 | 0.0642347 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | room_layout | -0.0210284 | -0.0961371 | 0.0540803 | t | -0.555742 | 0.57968 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_semantic | -0.0833086 | -0.158417 | -0.00819988 | t | -2.20169 | 0.0300826 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup25d | -0.0679158 | -0.143025 | 0.00719284 | t | -1.79489 | 0.0758181 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.0752117 | 0.000103056 | 0.15032 | t | 1.98771 | 0.0496927 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | vanishing_point | -0.284061 | -0.359169 | -0.208952 | t | -7.5072 | 3.08922e-11 | ols_fixed_effects | 125 |
| contrastive_self_supervised | crsa | Contrastive | 0.161109 | 0.111906 | 0.210313 | t | 6.59906 | 4.43496e-08 | ols_fixed_effects | 50 |
| contrastive_self_supervised | wrsa | Contrastive | 0.0699565 | 0.00922707 | 0.130686 | t | 2.32158 | 0.0249489 | ols_fixed_effects | 50 |
| language_alignment | crsa | CLIP | -0.137942 | -0.2439 | -0.0319848 | t | -2.63549 | 0.012095 | ols_fixed_effects | 45 |
| language_alignment | crsa | SLIP | 0.0331429 | -0.0728145 | 0.1391 | t | 0.633219 | 0.530383 | ols_fixed_effects | 45 |
| language_alignment | wrsa | CLIP | 0.0376268 | 0.00643492 | 0.0688187 | t | 2.44203 | 0.0193691 | ols_fixed_effects | 45 |
| language_alignment | wrsa | SLIP | 0.00727185 | -0.0239201 | 0.0384638 | t | 0.471952 | 0.63966 | ols_fixed_effects | 45 |
| imagenet_size | crsa | imagenet21k | -0.018074 | -0.0439763 | 0.00782831 | t | -1.38387 | 0.169389 | ols_fixed_effects_fallback | 120 |
| imagenet_size | wrsa | imagenet21k | -0.0177325 | -0.0336562 | -0.00180889 | t | -2.20856 | 0.0294217 | ols_fixed_effects_fallback | 120 |
| objects_faces_places | crsa | openimages | -0.0154399 | -0.107667 | 0.0767874 | t | -0.364757 | 0.72164 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | places256 | -0.0448698 | -0.137097 | 0.0473575 | t | -1.06002 | 0.310005 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | vggface2 | 0.206875 | 0.114648 | 0.299102 | t | 4.8873 | 0.000373881 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | openimages | 0.0213712 | -0.0556949 | 0.0984373 | t | 0.604207 | 0.556957 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | places256 | 0.0120447 | -0.0650214 | 0.0891108 | t | 0.340528 | 0.739346 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | vggface2 | 0.195761 | 0.118695 | 0.272827 | t | 5.53455 | 0.000128919 | ols_fixed_effects | 20 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.30152771472930906 | 105 | 0.14205844954121855 | 10 | 95 | 12 | 0.00979761148689664 | wrsa | 117 | alexnet_gn_ipcl_vggface2 | 0.4712762176990509 | pure_python_grid_search_two_breakpoints | not_computed |
