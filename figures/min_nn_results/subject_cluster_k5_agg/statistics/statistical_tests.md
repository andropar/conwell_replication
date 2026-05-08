# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.0431178 | -0.0538189 | -0.0324167 | t | -7.93361 | 6.07367e-14 | ols_fixed_effects | 270 |
| architecture | wrsa | Transformer | 0.020532 | 0.0111185 | 0.0299454 | t | 4.29461 | 2.46105e-05 | ols_fixed_effects | 270 |
| taskonomy_tasks | crsa | autoencoding | -0.0447908 | -0.0589709 | -0.0306107 | t | -6.26998 | 1.0295e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_object | 0.049243 | 0.0350628 | 0.0634231 | t | 6.89321 | 5.74013e-10 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_scene | 0.0201434 | 0.00596327 | 0.0343235 | t | 2.81974 | 0.00583877 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | curvature | 0.0171685 | 0.00298842 | 0.0313487 | t | 2.40331 | 0.0181684 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_euclidean | 0.0166047 | 0.00242453 | 0.0307848 | t | 2.32438 | 0.0222133 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_zbuffer | 0.0392971 | 0.025117 | 0.0534772 | t | 5.50095 | 3.12587e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_occlusion | -0.00913578 | -0.0233159 | 0.00504435 | t | -1.27886 | 0.20403 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_texture | 0.120739 | 0.106559 | 0.134919 | t | 16.9015 | 1.58357e-30 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | egomotion | 0.0106688 | -0.00351128 | 0.024849 | t | 1.49346 | 0.138595 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | fixated_pose | 0.00781787 | -0.00636225 | 0.021998 | t | 1.09437 | 0.27653 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | inpainting | -0.03797 | -0.0521501 | -0.0237898 | t | -5.31517 | 6.91778e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | jigsaw | 0.0497384 | 0.0355583 | 0.0639185 | t | 6.96256 | 4.1407e-10 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints2d | 0.0418358 | 0.0276557 | 0.056016 | t | 5.85633 | 6.6086e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints3d | 0.0582639 | 0.0440838 | 0.072444 | t | 8.15599 | 1.32721e-12 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | nonfixated_pose | 0.0259359 | 0.0117558 | 0.040116 | t | 3.6306 | 0.000455874 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | normal | 0.0222375 | 0.00805735 | 0.0364176 | t | 3.11288 | 0.00244138 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | point_matching | 0.0741282 | 0.0599481 | 0.0883083 | t | 10.3767 | 2.31247e-17 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | random_weights | -0.0124541 | -0.0266342 | 0.00172605 | t | -1.74337 | 0.0844713 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | reshading | 0.0561013 | 0.0419211 | 0.0702814 | t | 7.85325 | 5.80017e-12 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | room_layout | -0.0147548 | -0.0289349 | -0.0005747 | t | -2.06543 | 0.041577 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_semantic | 0.009079 | -0.00510113 | 0.0232591 | t | 1.27091 | 0.206832 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup25d | 0.0262222 | 0.0120421 | 0.0404023 | t | 3.67067 | 0.000397562 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.0946212 | 0.080441 | 0.108801 | t | 13.2454 | 2.16104e-23 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | vanishing_point | -0.0767311 | -0.0909112 | -0.062551 | t | -10.7411 | 3.84036e-18 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | autoencoding | -0.168108 | -0.205663 | -0.130553 | t | -8.88547 | 3.68285e-14 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_object | 0.0959496 | 0.0583949 | 0.133504 | t | 5.07149 | 1.92149e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_scene | 0.113181 | 0.0756261 | 0.150736 | t | 5.98226 | 3.77255e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | curvature | -0.12586 | -0.163414 | -0.088305 | t | -6.65241 | 1.77015e-09 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_euclidean | 0.0688276 | 0.0312729 | 0.106382 | t | 3.63794 | 0.000444628 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_zbuffer | 0.0730692 | 0.0355145 | 0.110624 | t | 3.86213 | 0.000204022 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_occlusion | 0.0734144 | 0.0358596 | 0.110969 | t | 3.88037 | 0.000191242 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_texture | 0.0310348 | -0.00651995 | 0.0685895 | t | 1.64037 | 0.104202 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | egomotion | 0.0394877 | 0.00193293 | 0.0770424 | t | 2.08715 | 0.0395239 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | fixated_pose | 0.0455581 | 0.00800332 | 0.0831128 | t | 2.40801 | 0.01795 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | inpainting | -0.00774964 | -0.0453044 | 0.0298051 | t | -0.409613 | 0.683002 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | jigsaw | -0.0122629 | -0.0498176 | 0.0252919 | t | -0.648163 | 0.518427 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints2d | -0.0391961 | -0.0767508 | -0.00164136 | t | -2.07174 | 0.0409715 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints3d | 0.0932453 | 0.0556906 | 0.1308 | t | 4.92856 | 3.45865e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | nonfixated_pose | -0.00822949 | -0.0457842 | 0.0293252 | t | -0.434976 | 0.664556 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | normal | 0.0384116 | 0.000856858 | 0.0759663 | t | 2.03027 | 0.0450951 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | point_matching | 0.0670944 | 0.0295397 | 0.104649 | t | 3.54633 | 0.00060601 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | random_weights | -0.0821787 | -0.119733 | -0.044624 | t | -4.34362 | 3.47806e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | reshading | 0.0758244 | 0.0382697 | 0.113379 | t | 4.00776 | 0.000121096 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | room_layout | 0.0602592 | 0.0227045 | 0.097814 | t | 3.18505 | 0.00195229 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_semantic | 0.0706511 | 0.0330964 | 0.108206 | t | 3.73432 | 0.000319252 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup25d | 0.0707396 | 0.0331849 | 0.108294 | t | 3.739 | 0.000314114 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.0925403 | 0.0549855 | 0.130095 | t | 4.89129 | 4.02563e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | vanishing_point | -0.128354 | -0.165909 | -0.0907991 | t | -6.78424 | 9.57034e-10 | ols_fixed_effects | 125 |
| contrastive_self_supervised | crsa | Contrastive | 0.0760956 | 0.0589767 | 0.0932146 | t | 8.95852 | 1.77286e-11 | ols_fixed_effects | 50 |
| contrastive_self_supervised | wrsa | Contrastive | 0.0561873 | 0.0421425 | 0.0702321 | t | 8.06264 | 3.27978e-10 | ols_fixed_effects | 50 |
| language_alignment | crsa | CLIP | -0.0834559 | -0.112407 | -0.0545042 | t | -5.83551 | 9.57219e-07 | ols_fixed_effects | 45 |
| language_alignment | crsa | SLIP | -0.00374875 | -0.0327004 | 0.0252029 | t | -0.262125 | 0.794641 | ols_fixed_effects | 45 |
| language_alignment | wrsa | CLIP | -0.0216769 | -0.0352706 | -0.00808327 | t | -3.22817 | 0.00256821 | ols_fixed_effects | 45 |
| language_alignment | wrsa | SLIP | -0.00062166 | -0.0142153 | 0.012972 | t | -0.0925789 | 0.926724 | ols_fixed_effects | 45 |
| imagenet_size | crsa | imagenet21k | -7.82659e-06 | -0.00817943 | 0.00816378 | z | -0.00187721 | 0.998502 | mixedlm_random_intercept | 120 |
| imagenet_size | wrsa | imagenet21k | -0.0159443 | -0.0246532 | -0.00723545 | t | -3.63099 | 0.000441806 | ols_fixed_effects_fallback | 120 |
| objects_faces_places | crsa | openimages | -0.021362 | -0.034961 | -0.00776294 | t | -3.42258 | 0.00505443 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | places256 | -0.0619267 | -0.0755257 | -0.0483277 | t | -9.92181 | 3.89809e-07 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | vggface2 | 0.00661885 | -0.00698016 | 0.0202179 | t | 1.06046 | 0.309812 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | openimages | -0.0216167 | -0.0379879 | -0.00524558 | t | -2.87694 | 0.0139094 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | places256 | -0.0527911 | -0.0691622 | -0.0364199 | t | -7.02589 | 1.38334e-05 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | vggface2 | -0.0597981 | -0.0761693 | -0.043427 | t | -7.95845 | 3.96565e-06 | ols_fixed_effects | 20 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 79 | 0.3517485809326172 | 107 | 0.2016326668858528 | 79 | 28 | 10 | 0.006287652190159355 | wrsa | 117 | ResNet50/BarlowTwins/BS2048_selfsupervised | 0.4486710798740387 | pure_python_grid_search_two_breakpoints | not_computed |
