# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.0730615 | -0.0860296 | -0.0600935 | t | -11.0932 | 1.00284e-23 | ols_fixed_effects | 270 |
| architecture | wrsa | Transformer | 0.0134507 | 0.00352732 | 0.0233741 | t | 2.66888 | 0.00808186 | ols_fixed_effects | 270 |
| taskonomy_tasks | crsa | autoencoding | -0.0732381 | -0.10414 | -0.042336 | t | -4.70442 | 8.53721e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_object | -0.0255503 | -0.0564524 | 0.00535178 | t | -1.64121 | 0.104025 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_scene | -0.0353358 | -0.0662379 | -0.00443375 | t | -2.26978 | 0.0254577 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | curvature | -0.0432276 | -0.0741297 | -0.0123255 | t | -2.77671 | 0.00660314 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_euclidean | -0.0487569 | -0.079659 | -0.0178548 | t | -3.13188 | 0.00230261 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_zbuffer | -0.0243709 | -0.055273 | 0.00653122 | t | -1.56545 | 0.120767 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_occlusion | -0.0478087 | -0.0787108 | -0.0169066 | t | -3.07097 | 0.00277541 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_texture | 0.0720917 | 0.0411896 | 0.102994 | t | 4.63078 | 1.14302e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | egomotion | -0.0198639 | -0.050766 | 0.0110382 | t | -1.27595 | 0.205052 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | fixated_pose | -0.0459528 | -0.0768549 | -0.0150507 | t | -2.95176 | 0.00397128 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | inpainting | -0.0483606 | -0.0792627 | -0.0174585 | t | -3.10643 | 0.00249026 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | jigsaw | -0.00271999 | -0.0336221 | 0.0281821 | t | -0.174718 | 0.861669 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints2d | -0.0542195 | -0.0851216 | -0.0233174 | t | -3.48277 | 0.000748964 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints3d | -0.00941802 | -0.0403201 | 0.0214841 | t | -0.604963 | 0.546631 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | nonfixated_pose | -0.00134104 | -0.0322431 | 0.0295611 | t | -0.0861415 | 0.931533 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | normal | -0.0408641 | -0.0717662 | -0.00996199 | t | -2.62489 | 0.010086 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | point_matching | 0.041451 | 0.0105489 | 0.0723531 | t | 2.66259 | 0.00909286 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | random_weights | -0.049983 | -0.0808851 | -0.0190809 | t | -3.21064 | 0.00180202 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | reshading | 0.0228103 | -0.00809175 | 0.0537124 | t | 1.46521 | 0.14613 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | room_layout | -0.0657283 | -0.0966304 | -0.0348262 | t | -4.22203 | 5.50171e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_semantic | -0.0455613 | -0.0764634 | -0.0146592 | t | -2.92661 | 0.00427779 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup25d | -0.00544396 | -0.0363461 | 0.0254581 | t | -0.349691 | 0.727337 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.0694512 | 0.0385491 | 0.100353 | t | 4.46117 | 2.21643e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | vanishing_point | -0.129305 | -0.160207 | -0.0984025 | t | -8.30583 | 6.37524e-13 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | autoencoding | -0.213042 | -0.259185 | -0.1669 | t | -9.1647 | 9.27163e-15 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_object | 0.105082 | 0.0589392 | 0.151225 | t | 4.52044 | 1.76135e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_scene | 0.10396 | 0.0578169 | 0.150103 | t | 4.47216 | 2.12424e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | curvature | -0.145767 | -0.19191 | -0.0996245 | t | -6.27065 | 1.02636e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_euclidean | 0.0942141 | 0.0480712 | 0.140357 | t | 4.05292 | 0.000102759 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_zbuffer | 0.0647415 | 0.0185986 | 0.110884 | t | 2.78506 | 0.00644801 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_occlusion | 0.0629217 | 0.0167788 | 0.109065 | t | 2.70678 | 0.00804215 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_texture | 0.00984962 | -0.0362933 | 0.0559925 | t | 0.423713 | 0.672723 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | egomotion | -0.0198102 | -0.0659531 | 0.0263327 | t | -0.8522 | 0.396224 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | fixated_pose | -0.0201745 | -0.0663175 | 0.0259684 | t | -0.867871 | 0.387629 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | inpainting | -0.0721152 | -0.118258 | -0.0259723 | t | -3.10226 | 0.00252227 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | jigsaw | -0.0857748 | -0.131918 | -0.0396318 | t | -3.68987 | 0.000372198 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints2d | -0.0610236 | -0.107167 | -0.0148807 | t | -2.62512 | 0.0100795 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints3d | 0.129443 | 0.0832998 | 0.175586 | t | 5.56839 | 2.3354e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | nonfixated_pose | -0.0862823 | -0.132425 | -0.0401394 | t | -3.71171 | 0.000345225 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | normal | 0.0627289 | 0.0165859 | 0.108872 | t | 2.69848 | 0.00823057 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | point_matching | 0.0320131 | -0.0141298 | 0.078156 | t | 1.37714 | 0.17167 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | random_weights | -0.101738 | -0.147881 | -0.0555951 | t | -4.37659 | 3.06739e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | reshading | 0.100107 | 0.0539641 | 0.14625 | t | 4.30642 | 4.00518e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | room_layout | 0.055968 | 0.00982512 | 0.102111 | t | 2.40764 | 0.0179668 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_semantic | 0.0564334 | 0.0102905 | 0.102576 | t | 2.42766 | 0.0170604 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup25d | 0.0657514 | 0.0196084 | 0.111894 | t | 2.8285 | 0.00569348 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.140939 | 0.0947959 | 0.187082 | t | 6.06293 | 2.62753e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | vanishing_point | -0.12814 | -0.174283 | -0.0819968 | t | -5.51234 | 2.97605e-07 | ols_fixed_effects | 125 |
| contrastive_self_supervised | crsa | Contrastive | 0.100653 | 0.0783476 | 0.122958 | t | 9.09441 | 1.14744e-11 | ols_fixed_effects | 50 |
| contrastive_self_supervised | wrsa | Contrastive | 0.0833226 | 0.0694519 | 0.0971934 | t | 12.1064 | 1.33942e-15 | ols_fixed_effects | 50 |
| language_alignment | crsa | CLIP | -0.0168228 | -0.0534214 | 0.0197759 | t | -0.930525 | 0.357974 | ols_fixed_effects | 45 |
| language_alignment | crsa | SLIP | 0.0180742 | -0.0185244 | 0.0546728 | t | 0.999745 | 0.323758 | ols_fixed_effects | 45 |
| language_alignment | wrsa | CLIP | -0.0101384 | -0.0267202 | 0.00644339 | t | -1.23775 | 0.223402 | ols_fixed_effects | 45 |
| language_alignment | wrsa | SLIP | 0.00339201 | -0.0131898 | 0.0199738 | t | 0.414114 | 0.681119 | ols_fixed_effects | 45 |
| imagenet_size | crsa | imagenet21k | 0.00581469 | -0.00337157 | 0.0150009 | t | 1.25536 | 0.212189 | ols_fixed_effects_fallback | 120 |
| imagenet_size | wrsa | imagenet21k | -0.00754761 | -0.0165076 | 0.00141233 | z | -1.65102 | 0.0987343 | mixedlm_random_intercept | 120 |
| objects_faces_places | crsa | openimages | -0.0243098 | -0.0447776 | -0.003842 | t | -2.5878 | 0.0237541 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | places256 | -0.0768111 | -0.0972789 | -0.0563433 | t | -8.1766 | 3.0046e-06 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | vggface2 | -0.00342742 | -0.0238952 | 0.0170404 | t | -0.364852 | 0.721571 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | openimages | -0.027304 | -0.0963356 | 0.0417276 | t | -0.861783 | 0.405699 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | places256 | -0.102274 | -0.171306 | -0.0332424 | t | -3.22803 | 0.00724575 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | vggface2 | -0.0995069 | -0.168539 | -0.0304753 | t | -3.14069 | 0.00852097 | ols_fixed_effects | 20 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 84 | 0.39011054039001464 | 101 | 0.19635679423809047 | 84 | 17 | 16 | 0.009701087034044754 | wrsa | 117 | levit_128_classification | 0.5260837495326995 | pure_python_grid_search_two_breakpoints | not_computed |
