# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.0599055 | -0.0716509 | -0.0481601 | t | -10.0425 | 2.61851e-20 | ols_fixed_effects | 270 |
| architecture | wrsa | Transformer | 0.0155378 | 0.00685634 | 0.0242192 | t | 3.52404 | 0.000500785 | ols_fixed_effects | 270 |
| taskonomy_tasks | crsa | autoencoding | -0.0325008 | -0.0524459 | -0.0125557 | t | -3.23456 | 0.00167139 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_object | 0.042781 | 0.0228359 | 0.0627261 | t | 4.25767 | 4.8136e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_scene | 0.0274705 | 0.00752537 | 0.0474156 | t | 2.73393 | 0.00745262 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | curvature | 0.0167597 | -0.00318535 | 0.0367048 | t | 1.66797 | 0.0985804 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_euclidean | 0.0186648 | -0.00128032 | 0.0386099 | t | 1.85756 | 0.0662963 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_zbuffer | 0.0390109 | 0.0190658 | 0.058956 | t | 3.88246 | 0.000189831 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_occlusion | -0.00853897 | -0.0284841 | 0.0114061 | t | -0.849819 | 0.39754 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_texture | 0.100803 | 0.0808583 | 0.120749 | t | 10.0322 | 1.26829e-16 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | egomotion | 0.0119916 | -0.00795346 | 0.0319367 | t | 1.19344 | 0.235639 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | fixated_pose | -0.00204063 | -0.0219857 | 0.0179045 | t | -0.203089 | 0.839496 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | inpainting | -0.0217477 | -0.0416928 | -0.00180263 | t | -2.16439 | 0.032915 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | jigsaw | 0.0247323 | 0.00478722 | 0.0446774 | t | 2.46142 | 0.0156238 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints2d | 0.0333227 | 0.0133776 | 0.0532678 | t | 3.31635 | 0.00128851 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints3d | 0.0528266 | 0.0328815 | 0.0727717 | t | 5.25743 | 8.83128e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | nonfixated_pose | 0.0145395 | -0.00540558 | 0.0344846 | t | 1.44701 | 0.151152 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | normal | 0.0360146 | 0.0160695 | 0.0559597 | t | 3.58426 | 0.000533418 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | point_matching | 0.0686464 | 0.0487013 | 0.0885915 | t | 6.83186 | 7.65673e-10 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | random_weights | -0.0364547 | -0.0563998 | -0.0165096 | t | -3.62806 | 0.000459836 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | reshading | 0.0382012 | 0.0182561 | 0.0581463 | t | 3.80188 | 0.000252269 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | room_layout | -0.00916132 | -0.0291064 | 0.0107838 | t | -0.911757 | 0.364181 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_semantic | 0.0305355 | 0.0105904 | 0.0504806 | t | 3.03897 | 0.00305852 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup25d | 0.0351649 | 0.0152198 | 0.05511 | t | 3.4997 | 0.000708053 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.0807307 | 0.0607857 | 0.100676 | t | 8.03452 | 2.4012e-12 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | vanishing_point | -0.0533121 | -0.0732572 | -0.033367 | t | -5.30575 | 7.19972e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | autoencoding | -0.153372 | -0.205296 | -0.101448 | t | -5.86322 | 6.40963e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_object | 0.139134 | 0.0872098 | 0.191057 | t | 5.31891 | 6.80897e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_scene | 0.156914 | 0.104991 | 0.208838 | t | 5.99864 | 3.50589e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | curvature | -0.0984895 | -0.150413 | -0.0465656 | t | -3.76513 | 0.000286836 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_euclidean | 0.122066 | 0.0701427 | 0.17399 | t | 4.66645 | 9.92657e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_zbuffer | 0.119589 | 0.0676656 | 0.171513 | t | 4.57176 | 1.44154e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_occlusion | 0.0876284 | 0.0357046 | 0.139552 | t | 3.34993 | 0.00115655 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_texture | 0.0132117 | -0.0387121 | 0.0651355 | t | 0.505068 | 0.614669 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | egomotion | 0.0719781 | 0.0200543 | 0.123902 | t | 2.75164 | 0.0070896 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | fixated_pose | 0.0581146 | 0.00619074 | 0.110038 | t | 2.22165 | 0.0286565 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | inpainting | 0.0184714 | -0.0334525 | 0.0703952 | t | 0.706137 | 0.481813 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | jigsaw | -0.0195804 | -0.0715042 | 0.0323434 | t | -0.748534 | 0.455968 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints2d | -0.0780234 | -0.129947 | -0.0260996 | t | -2.98274 | 0.00362159 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints3d | 0.129802 | 0.0778786 | 0.181726 | t | 4.96219 | 3.0143e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | nonfixated_pose | 0.0202707 | -0.0316532 | 0.0721945 | t | 0.774923 | 0.44029 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | normal | 0.0396316 | -0.0122922 | 0.0915554 | t | 1.51507 | 0.133041 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | point_matching | 0.0928301 | 0.0409062 | 0.144754 | t | 3.54878 | 0.000601044 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | random_weights | -0.0484378 | -0.100362 | 0.00348599 | t | -1.85172 | 0.0671398 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | reshading | 0.114332 | 0.0624085 | 0.166256 | t | 4.37078 | 3.1361e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | room_layout | 0.0904255 | 0.0385017 | 0.142349 | t | 3.45686 | 0.00081592 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_semantic | 0.11508 | 0.0631561 | 0.167004 | t | 4.39936 | 2.81139e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup25d | 0.102009 | 0.0500847 | 0.153932 | t | 3.89966 | 0.000178562 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.0962892 | 0.0443654 | 0.148213 | t | 3.68102 | 0.000383699 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | vanishing_point | -0.12996 | -0.181884 | -0.0780366 | t | -4.96823 | 2.9406e-06 | ols_fixed_effects | 125 |
| contrastive_self_supervised | crsa | Contrastive | 0.0650715 | 0.0464771 | 0.0836659 | t | 7.05282 | 9.5659e-09 | ols_fixed_effects | 50 |
| contrastive_self_supervised | wrsa | Contrastive | 0.0557853 | 0.040268 | 0.0713026 | t | 7.24533 | 5.00291e-09 | ols_fixed_effects | 50 |
| language_alignment | crsa | CLIP | -0.0287151 | -0.057327 | -0.00010327 | t | -2.0317 | 0.0492193 | ols_fixed_effects | 45 |
| language_alignment | crsa | SLIP | 0.0144075 | -0.0142044 | 0.0430193 | t | 1.01938 | 0.314468 | ols_fixed_effects | 45 |
| language_alignment | wrsa | CLIP | -0.0239835 | -0.0369075 | -0.0110595 | t | -3.75674 | 0.000576982 | ols_fixed_effects | 45 |
| language_alignment | wrsa | SLIP | 0.00699237 | -0.00593164 | 0.0199164 | t | 1.09527 | 0.280291 | ols_fixed_effects | 45 |
| imagenet_size | crsa | imagenet21k | 0.0120161 | 0.00256013 | 0.021472 | t | 2.52022 | 0.0132632 | ols_fixed_effects_fallback | 120 |
| imagenet_size | wrsa | imagenet21k | -0.00851356 | -0.0152721 | -0.001755 | t | -2.49826 | 0.0140622 | ols_fixed_effects_fallback | 120 |
| objects_faces_places | crsa | openimages | -0.0139742 | -0.0303914 | 0.00244295 | t | -1.8546 | 0.0883799 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | places256 | -0.075147 | -0.0915641 | -0.0587298 | t | -9.97317 | 3.68679e-07 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | vggface2 | -0.0106296 | -0.0270468 | 0.00578758 | t | -1.41071 | 0.183724 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | openimages | -0.0155614 | -0.0525192 | 0.0213965 | t | -0.917405 | 0.377 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | places256 | -0.044401 | -0.0813588 | -0.00744313 | t | -2.61762 | 0.0224836 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | vggface2 | -0.114626 | -0.151584 | -0.0776686 | t | -6.75769 | 2.02153e-05 | ols_fixed_effects | 20 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 91 | 0.39347697496414186 | 107 | 0.20046336203813547 | 91 | 16 | 10 | 0.008233364627382514 | wrsa | 117 | swin_tiny_patch4_window7_224_classification | 0.5232748031616211 | pure_python_grid_search_two_breakpoints | not_computed |
