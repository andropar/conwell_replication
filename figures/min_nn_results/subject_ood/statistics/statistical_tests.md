# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.0713776 | -0.0987124 | -0.0440427 | t | -5.14149 | 5.3141e-07 | ols_fixed_effects | 270 |
| architecture | wrsa | Transformer | 0.0303284 | 0.0154523 | 0.0452044 | t | 4.01425 | 7.77566e-05 | ols_fixed_effects | 270 |
| taskonomy_tasks | crsa | autoencoding | -0.0966111 | -0.134215 | -0.0590069 | t | -5.09974 | 1.70898e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_object | -0.104521 | -0.142125 | -0.0669171 | t | -5.51729 | 2.91311e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_scene | -0.156156 | -0.19376 | -0.118552 | t | -8.2429 | 8.6767e-13 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | curvature | -0.103761 | -0.141365 | -0.0661567 | t | -5.47715 | 3.46317e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_euclidean | -0.149571 | -0.187175 | -0.111967 | t | -7.89528 | 4.72905e-12 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_zbuffer | -0.111771 | -0.149375 | -0.0741667 | t | -5.89997 | 5.44471e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_occlusion | -0.128301 | -0.165905 | -0.0906965 | t | -6.77251 | 1.01099e-09 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_texture | 0.0618583 | 0.0242542 | 0.0994625 | t | 3.26527 | 0.00151659 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | egomotion | -0.0680069 | -0.105611 | -0.0304028 | t | -3.58983 | 0.000523465 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | fixated_pose | -0.0721816 | -0.109786 | -0.0345774 | t | -3.8102 | 0.000245011 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | inpainting | 0.1148 | 0.077196 | 0.152404 | t | 6.05987 | 2.6639e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | jigsaw | 0.000397232 | -0.0372069 | 0.0380014 | t | 0.0209684 | 0.983314 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints2d | 0.00939419 | -0.02821 | 0.0469983 | t | 0.495885 | 0.621109 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints3d | -0.0350714 | -0.0726756 | 0.00253273 | t | -1.85129 | 0.0672019 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | nonfixated_pose | -0.0337656 | -0.0713698 | 0.00383852 | t | -1.78236 | 0.0778514 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | normal | -0.0351235 | -0.0727277 | 0.00248062 | t | -1.85404 | 0.0668036 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | point_matching | 0.071843 | 0.0342388 | 0.109447 | t | 3.79232 | 0.000260853 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | random_weights | 0.235217 | 0.197613 | 0.272821 | t | 12.4162 | 1.11281e-21 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | reshading | -0.0882621 | -0.125866 | -0.0506579 | t | -4.65903 | 1.02228e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | room_layout | -0.211096 | -0.2487 | -0.173491 | t | -11.143 | 5.34253e-19 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_semantic | -0.0569153 | -0.0945194 | -0.0193111 | t | -3.00435 | 0.00339473 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup25d | -0.0832361 | -0.12084 | -0.045632 | t | -4.39373 | 2.87276e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.0292501 | -0.00835407 | 0.0668542 | t | 1.544 | 0.125878 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | vanishing_point | -0.182921 | -0.220525 | -0.145316 | t | -9.6557 | 8.16874e-16 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | autoencoding | -0.155602 | -0.242263 | -0.0689406 | t | -3.56408 | 0.000570941 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_object | -0.0677815 | -0.154443 | 0.0188796 | t | -1.55255 | 0.123822 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_scene | 0.0121093 | -0.0745517 | 0.0987703 | t | 0.277365 | 0.782096 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | curvature | -0.198692 | -0.285353 | -0.112031 | t | -4.55108 | 1.56299e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_euclidean | -0.0915806 | -0.178242 | -0.00491952 | t | -2.09767 | 0.0385616 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_zbuffer | -0.0495314 | -0.136192 | 0.0371296 | t | -1.13452 | 0.2594 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_occlusion | -0.004824 | -0.091485 | 0.081837 | t | -0.110494 | 0.912248 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_texture | 0.0698044 | -0.0168566 | 0.156465 | t | 1.59888 | 0.113133 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | egomotion | -0.0188894 | -0.10555 | 0.0677717 | t | -0.432664 | 0.666229 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | fixated_pose | -0.12348 | -0.210141 | -0.0368191 | t | -2.82833 | 0.00569631 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | inpainting | 0.205435 | 0.118774 | 0.292096 | t | 4.70553 | 8.4996e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | jigsaw | -0.131413 | -0.218074 | -0.0447516 | t | -3.01003 | 0.00333735 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints2d | -0.163067 | -0.249728 | -0.0764058 | t | -3.73507 | 0.00031842 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints3d | 0.158292 | 0.0716312 | 0.244953 | t | 3.62571 | 0.000463529 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | nonfixated_pose | -0.21217 | -0.298831 | -0.125509 | t | -4.85978 | 4.57462e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | normal | -0.0389306 | -0.125592 | 0.0477304 | t | -0.891711 | 0.374777 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | point_matching | -0.0298258 | -0.116487 | 0.0568352 | t | -0.683164 | 0.496148 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | random_weights | -0.025024 | -0.111685 | 0.061637 | t | -0.57318 | 0.567863 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | reshading | 0.043279 | -0.043382 | 0.12994 | t | 0.991312 | 0.324025 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | room_layout | -0.0113964 | -0.0980574 | 0.0752647 | t | -0.261036 | 0.794624 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_semantic | -0.0887117 | -0.175373 | -0.00205061 | t | -2.03195 | 0.0449214 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup25d | 0.00628762 | -0.0803734 | 0.0929487 | t | 0.144019 | 0.885788 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.112982 | 0.0263212 | 0.199643 | t | 2.58787 | 0.0111555 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | vanishing_point | -0.237757 | -0.324418 | -0.151096 | t | -5.44587 | 3.96132e-07 | ols_fixed_effects | 125 |
| contrastive_self_supervised | crsa | Contrastive | 0.159009 | 0.115799 | 0.20222 | t | 7.41628 | 2.81827e-09 | ols_fixed_effects | 50 |
| contrastive_self_supervised | wrsa | Contrastive | 0.0507487 | -0.00886356 | 0.110361 | t | 1.71571 | 0.09325 | ols_fixed_effects | 50 |
| language_alignment | crsa | CLIP | -0.156977 | -0.25487 | -0.059084 | t | -3.24623 | 0.00244426 | ols_fixed_effects | 45 |
| language_alignment | crsa | SLIP | -0.000153042 | -0.0980459 | 0.0977398 | t | -0.00316487 | 0.997491 | ols_fixed_effects | 45 |
| language_alignment | wrsa | CLIP | 0.0352993 | 0.00554227 | 0.0650563 | t | 2.40144 | 0.0213285 | ols_fixed_effects | 45 |
| language_alignment | wrsa | SLIP | -0.0201505 | -0.0499076 | 0.00960646 | t | -1.37086 | 0.178462 | ols_fixed_effects | 45 |
| imagenet_size | crsa | imagenet21k | -0.0409189 | -0.0668592 | -0.0149785 | t | -3.12845 | 0.0022856 | ols_fixed_effects_fallback | 120 |
| imagenet_size | wrsa | imagenet21k | -0.0285014 | -0.0472899 | -0.00971284 | t | -3.00852 | 0.00330007 | ols_fixed_effects_fallback | 120 |
| objects_faces_places | crsa | openimages | -0.015418 | -0.106106 | 0.0752703 | t | -0.370422 | 0.717524 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | places256 | -0.0485976 | -0.139286 | 0.0420907 | t | -1.16757 | 0.265656 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | vggface2 | 0.211232 | 0.120543 | 0.30192 | t | 5.0749 | 0.000272956 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | openimages | 0.0554167 | -0.0294286 | 0.140262 | t | 1.42309 | 0.180184 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | places256 | 0.0770484 | -0.00779693 | 0.161894 | t | 1.97859 | 0.0712792 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | vggface2 | 0.197574 | 0.112729 | 0.28242 | t | 5.07367 | 0.000273518 | ols_fixed_effects | 20 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 0.32575114369392394 | 105 | 0.14902504831552504 | 11 | 94 | 12 | 0.008953629237480689 | wrsa | 117 | alexnet_gn_ipcl_vggface2 | 0.48647719621658325 | pure_python_grid_search_two_breakpoints | not_computed |
