# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.0873574 | -0.114837 | -0.0598781 | t | -6.25947 | 1.55496e-09 | ols_fixed_effects | 270 |
| architecture | wrsa | Transformer | 0.0465481 | 0.0362975 | 0.0567987 | t | 8.94119 | 6.87915e-17 | ols_fixed_effects | 270 |
| taskonomy_tasks | crsa | autoencoding | -0.031545 | -0.0611519 | -0.00193797 | t | -2.11491 | 0.0370271 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_object | 0.034014 | 0.00440703 | 0.063621 | t | 2.28045 | 0.0247929 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_scene | 0.0111154 | -0.0184916 | 0.0407224 | t | 0.745225 | 0.457957 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | curvature | -0.0617506 | -0.0913576 | -0.0321436 | t | -4.14004 | 7.46263e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_euclidean | -0.0550838 | -0.0846907 | -0.0254768 | t | -3.69306 | 0.000368139 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_zbuffer | 0.00599228 | -0.0236147 | 0.0355993 | t | 0.401749 | 0.688762 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_occlusion | -0.0247733 | -0.0543803 | 0.00483365 | t | -1.66091 | 0.0999933 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_texture | 0.191019 | 0.161412 | 0.220626 | t | 12.8068 | 1.72329e-22 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | egomotion | -0.0531952 | -0.0828022 | -0.0235883 | t | -3.56645 | 0.000566413 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | fixated_pose | -0.0530206 | -0.0826275 | -0.0234136 | t | -3.55474 | 0.000589152 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | inpainting | 0.11149 | 0.0818827 | 0.141097 | t | 7.47476 | 3.61058e-11 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | jigsaw | 0.0785725 | 0.0489655 | 0.108179 | t | 5.26785 | 8.45139e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints2d | 0.060344 | 0.030737 | 0.0899509 | t | 4.04573 | 0.000105489 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints3d | 0.0704165 | 0.0408095 | 0.100023 | t | 4.72104 | 7.99038e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | nonfixated_pose | -0.0643755 | -0.0939824 | -0.0347685 | t | -4.31602 | 3.86229e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | normal | 0.0276175 | -0.00198952 | 0.0572245 | t | 1.8516 | 0.0671573 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | point_matching | 0.146001 | 0.116394 | 0.175608 | t | 9.78856 | 4.23268e-16 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | random_weights | 0.156714 | 0.127107 | 0.186321 | t | 10.5068 | 1.21724e-17 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | reshading | -0.00154635 | -0.0311533 | 0.0280606 | t | -0.103675 | 0.917644 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | room_layout | -0.221672 | -0.251279 | -0.192065 | t | -14.8619 | 1.23723e-26 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_semantic | -0.0382773 | -0.0678843 | -0.00867035 | t | -2.56628 | 0.0118256 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup25d | 0.00354701 | -0.02606 | 0.033154 | t | 0.237807 | 0.812537 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.089167 | 0.05956 | 0.118774 | t | 5.97815 | 3.84245e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | vanishing_point | -0.127112 | -0.156719 | -0.0975047 | t | -8.52214 | 2.20521e-13 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | autoencoding | -0.376414 | -0.431733 | -0.321095 | t | -13.5067 | 6.33622e-24 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_object | 0.0513658 | -0.00395335 | 0.106685 | t | 1.84313 | 0.0683958 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_scene | 0.0544652 | -0.000853871 | 0.109784 | t | 1.95435 | 0.05357 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | curvature | -0.457466 | -0.512785 | -0.402147 | t | -16.415 | 1.28066e-29 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_euclidean | 0.0140683 | -0.0412508 | 0.0693874 | t | 0.504806 | 0.614852 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_zbuffer | 0.0103843 | -0.0449348 | 0.0657034 | t | 0.372615 | 0.710257 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_occlusion | -0.079381 | -0.1347 | -0.0240619 | t | -2.84839 | 0.00537589 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_texture | -0.0637768 | -0.119096 | -0.00845771 | t | -2.28847 | 0.0243033 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | egomotion | -0.0045572 | -0.0598763 | 0.0507619 | t | -0.163524 | 0.87045 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | fixated_pose | 0.0115673 | -0.0437518 | 0.0668864 | t | 0.415062 | 0.679023 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | inpainting | -0.0540289 | -0.109348 | 0.00129017 | t | -1.93869 | 0.0554761 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | jigsaw | -0.0105011 | -0.0658202 | 0.044818 | t | -0.376805 | 0.70715 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints2d | -0.0343794 | -0.0896985 | 0.0209397 | t | -1.23362 | 0.220358 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints3d | 0.0620468 | 0.00672768 | 0.117366 | t | 2.22639 | 0.0283265 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | nonfixated_pose | -0.057594 | -0.112913 | -0.00227485 | t | -2.06661 | 0.0414633 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | normal | 0.0252353 | -0.0300838 | 0.0805544 | t | 0.905502 | 0.367466 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | point_matching | -0.00642664 | -0.0617457 | 0.0488925 | t | -0.230603 | 0.818114 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | random_weights | -0.0777411 | -0.13306 | -0.022422 | t | -2.78954 | 0.0063662 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | reshading | 0.0413183 | -0.0140008 | 0.0966374 | t | 1.4826 | 0.141456 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | room_layout | 0.0329533 | -0.0223658 | 0.0882724 | t | 1.18245 | 0.239949 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_semantic | 0.0342196 | -0.0210995 | 0.0895387 | t | 1.22788 | 0.222493 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup25d | 0.0515488 | -0.00377027 | 0.106868 | t | 1.8497 | 0.0674336 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.0789247 | 0.0236056 | 0.134244 | t | 2.83201 | 0.0056362 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | vanishing_point | -0.300101 | -0.35542 | -0.244782 | t | -10.7684 | 3.35847e-18 | ols_fixed_effects | 125 |
| contrastive_self_supervised | crsa | Contrastive | 0.0990565 | 0.0737008 | 0.124412 | t | 7.87338 | 6.13568e-10 | ols_fixed_effects | 50 |
| contrastive_self_supervised | wrsa | Contrastive | 0.00599895 | -0.017645 | 0.0296429 | t | 0.51134 | 0.611669 | ols_fixed_effects | 50 |
| language_alignment | crsa | CLIP | -0.171775 | -0.271384 | -0.072165 | t | -3.49102 | 0.00123603 | ols_fixed_effects | 45 |
| language_alignment | crsa | SLIP | 0.0284963 | -0.0711134 | 0.128106 | t | 0.579138 | 0.565913 | ols_fixed_effects | 45 |
| language_alignment | wrsa | CLIP | 0.0120048 | -0.00029948 | 0.0243091 | t | 1.97512 | 0.0555516 | ols_fixed_effects | 45 |
| language_alignment | wrsa | SLIP | 0.00445185 | -0.00785245 | 0.0167561 | t | 0.732451 | 0.468388 | ols_fixed_effects | 45 |
| imagenet_size | crsa | imagenet21k | 0.012709 | -0.00197867 | 0.0273967 | z | 1.69593 | 0.0899 | mixedlm_random_intercept | 120 |
| imagenet_size | wrsa | imagenet21k | 0.00118484 | -0.00696061 | 0.00933029 | t | 0.288486 | 0.773555 | ols_fixed_effects_fallback | 120 |
| objects_faces_places | crsa | openimages | -0.00393299 | -0.0401327 | 0.0322667 | t | -0.236721 | 0.816866 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | places256 | -0.0521761 | -0.0883758 | -0.0159764 | t | -3.14041 | 0.00852544 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | vggface2 | 0.116094 | 0.0798941 | 0.152294 | t | 6.98753 | 1.45963e-05 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | openimages | -0.0343968 | -0.0643662 | -0.00442752 | t | -2.5007 | 0.0278796 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | places256 | -0.0569285 | -0.0868978 | -0.0269592 | t | -4.13878 | 0.0013738 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | vggface2 | -0.057477 | -0.0874463 | -0.0275077 | t | -4.17866 | 0.00127955 | ols_fixed_effects | 20 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 75 | 0.5047031760215759 | 107 | 0.4284873366355896 | 75 | 32 | 10 | 0.06139135444037153 | wrsa | 117 | ViT/B/CLIP_slip | 0.5946556806564331 | pure_python_grid_search_two_breakpoints | not_computed |
