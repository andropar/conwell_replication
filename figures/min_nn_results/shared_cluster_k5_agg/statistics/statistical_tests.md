# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.0509171 | -0.0644157 | -0.0374184 | t | -7.42707 | 1.53736e-12 | ols_fixed_effects | 270 |
| architecture | wrsa | Transformer | 0.0320819 | 0.0223702 | 0.0417936 | t | 6.50443 | 3.88903e-10 | ols_fixed_effects | 270 |
| taskonomy_tasks | crsa | autoencoding | -0.0653037 | -0.0820448 | -0.0485626 | t | -7.74303 | 9.89782e-12 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_object | 0.0188646 | 0.00212354 | 0.0356057 | t | 2.23677 | 0.0276156 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_scene | -0.00891903 | -0.0256601 | 0.00782205 | t | -1.05753 | 0.292925 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | curvature | 0.00237502 | -0.0143661 | 0.0191161 | t | 0.281605 | 0.778852 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_euclidean | -0.0183973 | -0.0351384 | -0.00165622 | t | -2.18136 | 0.0315986 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_zbuffer | 0.0144539 | -0.00228718 | 0.031195 | t | 1.71379 | 0.0897939 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_occlusion | -0.0574458 | -0.0741869 | -0.0407047 | t | -6.81132 | 8.4304e-10 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_texture | 0.118243 | 0.101502 | 0.134984 | t | 14.02 | 5.8153e-25 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | egomotion | -0.0118731 | -0.0286142 | 0.00486794 | t | -1.40779 | 0.162422 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | fixated_pose | -0.0159447 | -0.0326858 | 0.000796358 | t | -1.89056 | 0.061699 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | inpainting | -0.0613902 | -0.0781313 | -0.0446491 | t | -7.27902 | 9.22104e-11 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | jigsaw | 0.0309055 | 0.0141645 | 0.0476466 | t | 3.66446 | 0.000406118 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints2d | 0.0640651 | 0.047324 | 0.0808062 | t | 7.59617 | 2.01256e-11 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints3d | 0.0253517 | 0.00861066 | 0.0420928 | t | 3.00595 | 0.00337847 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | nonfixated_pose | 0.00895714 | -0.00778395 | 0.0256982 | t | 1.06204 | 0.29088 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | normal | -0.00851338 | -0.0252545 | 0.00822771 | t | -1.00943 | 0.315307 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | point_matching | 0.0606543 | 0.0439132 | 0.0773954 | t | 7.19176 | 1.3977e-10 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | random_weights | -0.0276834 | -0.0444245 | -0.0109423 | t | -3.28241 | 0.00143614 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | reshading | 0.0347712 | 0.0180301 | 0.0515123 | t | 4.12281 | 7.95264e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | room_layout | -0.0421931 | -0.0589342 | -0.025452 | t | -5.00282 | 2.55123e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_semantic | -0.0207753 | -0.0375163 | -0.00403417 | t | -2.46331 | 0.0155465 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup25d | 0.00361037 | -0.0131307 | 0.0203515 | t | 0.42808 | 0.669551 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.102285 | 0.0855442 | 0.119026 | t | 12.1279 | 4.45045e-21 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | vanishing_point | -0.0904861 | -0.107227 | -0.073745 | t | -10.7289 | 4.07777e-18 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | autoencoding | -0.219034 | -0.261742 | -0.176326 | t | -10.1802 | 6.10147e-17 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_object | 0.0896257 | 0.0469176 | 0.132334 | t | 4.16561 | 6.78832e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_scene | 0.0909639 | 0.0482557 | 0.133672 | t | 4.22781 | 5.38412e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | curvature | -0.12125 | -0.163958 | -0.0785418 | t | -5.63544 | 1.74497e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_euclidean | 0.0417479 | -0.000960262 | 0.0844561 | t | 1.94035 | 0.0552709 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_zbuffer | 0.0600675 | 0.0173594 | 0.102776 | t | 2.79181 | 0.0063251 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_occlusion | 0.0325613 | -0.0101469 | 0.0752695 | t | 1.51338 | 0.133469 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_texture | 0.0256744 | -0.0170338 | 0.0683826 | t | 1.19329 | 0.235696 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | egomotion | -0.0090633 | -0.0517715 | 0.0336449 | t | -0.421243 | 0.674519 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | fixated_pose | 0.0228843 | -0.0198239 | 0.0655924 | t | 1.06361 | 0.290173 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | inpainting | -0.0500368 | -0.092745 | -0.00732866 | t | -2.3256 | 0.0221448 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | jigsaw | -0.0787004 | -0.121409 | -0.0359922 | t | -3.65782 | 0.000415442 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints2d | -0.0621493 | -0.104857 | -0.0194411 | t | -2.88857 | 0.00478312 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints3d | 0.108656 | 0.065948 | 0.151364 | t | 5.05011 | 2.09927e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | nonfixated_pose | -0.079351 | -0.122059 | -0.0366429 | t | -3.68807 | 0.00037452 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | normal | 0.0415742 | -0.00113394 | 0.0842824 | t | 1.93228 | 0.0562728 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | point_matching | 0.0352311 | -0.00747705 | 0.0779393 | t | 1.63747 | 0.104807 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | random_weights | -0.102283 | -0.144991 | -0.0595751 | t | -4.75391 | 7.00694e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | reshading | 0.07639 | 0.0336818 | 0.119098 | t | 3.55044 | 0.000597702 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | room_layout | 0.0349444 | -0.00776374 | 0.0776526 | t | 1.62414 | 0.107625 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_semantic | 0.0680311 | 0.0253229 | 0.110739 | t | 3.16194 | 0.00209797 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup25d | 0.0599323 | 0.0172241 | 0.10264 | t | 2.78552 | 0.0064395 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.13351 | 0.0908017 | 0.176218 | t | 6.20525 | 1.38167e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | vanishing_point | -0.139046 | -0.181754 | -0.0963379 | t | -6.46256 | 4.26161e-09 | ols_fixed_effects | 125 |
| contrastive_self_supervised | crsa | Contrastive | 0.0869142 | 0.0637234 | 0.110105 | t | 7.55316 | 1.78246e-09 | ols_fixed_effects | 50 |
| contrastive_self_supervised | wrsa | Contrastive | 0.0862019 | 0.0686876 | 0.103716 | t | 9.91924 | 8.57445e-13 | ols_fixed_effects | 50 |
| language_alignment | crsa | CLIP | -0.0770217 | -0.111272 | -0.0427711 | t | -4.5524 | 5.31625e-05 | ols_fixed_effects | 45 |
| language_alignment | crsa | SLIP | 0.00547504 | -0.0287756 | 0.0397256 | t | 0.323604 | 0.748012 | ols_fixed_effects | 45 |
| language_alignment | wrsa | CLIP | -0.0404609 | -0.0564261 | -0.0244958 | t | -5.13047 | 8.82641e-06 | ols_fixed_effects | 45 |
| language_alignment | wrsa | SLIP | -0.0205179 | -0.0364831 | -0.00455274 | t | -2.60168 | 0.0131499 | ols_fixed_effects | 45 |
| imagenet_size | crsa | imagenet21k | 0.00271048 | -0.00434914 | 0.00977009 | z | 0.752511 | 0.451744 | mixedlm_random_intercept | 120 |
| imagenet_size | wrsa | imagenet21k | -0.0107944 | -0.0213471 | -0.000241647 | t | -2.02868 | 0.0450714 | ols_fixed_effects_fallback | 120 |
| objects_faces_places | crsa | openimages | -0.0126078 | -0.0367222 | 0.0115066 | t | -1.13916 | 0.276868 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | places256 | -0.0574521 | -0.0815665 | -0.0333377 | t | -5.19098 | 0.000225221 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | vggface2 | 0.0230178 | -0.00109663 | 0.0471322 | t | 2.07973 | 0.0596516 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | openimages | 0.000272775 | -0.0401673 | 0.0407129 | t | 0.0146965 | 0.988516 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | places256 | -0.0503403 | -0.0907804 | -0.00990021 | t | -2.71221 | 0.018878 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | vggface2 | -0.00420533 | -0.0446454 | 0.0362348 | t | -0.226573 | 0.824569 | ols_fixed_effects | 20 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 82 | 0.31411941647529595 | 107 | 0.1991125780344009 | 82 | 25 | 10 | 0.005258068445231292 | wrsa | 117 | ResNet50/BarlowTwins/BS2048_selfsupervised | 0.42763388037681577 | pure_python_grid_search_two_breakpoints | not_computed |
