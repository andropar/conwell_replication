# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.0906151 | -0.113851 | -0.0673793 | t | -7.7326 | 6.71012e-12 | ols_fixed_effects | 108 |
| architecture | wrsa | Transformer | 0.0277633 | 0.0116926 | 0.043834 | t | 3.42545 | 0.000876977 | ols_fixed_effects | 108 |
| taskonomy_tasks | crsa | autoencoding | -0.0652804 | -0.0985434 | -0.0320175 | t | -4.05051 | 0.00046373 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | class_object | 0.0151324 | -0.0181306 | 0.0483954 | t | 0.938933 | 0.357119 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | class_scene | -0.0264871 | -0.0597501 | 0.00677586 | t | -1.64347 | 0.113324 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | curvature | -0.000283013 | -0.033546 | 0.03298 | t | -0.0175603 | 0.986135 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | depth_euclidean | -0.0147863 | -0.0480493 | 0.0184767 | t | -0.917458 | 0.368031 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | depth_zbuffer | -0.000915409 | -0.0341784 | 0.0323476 | t | -0.0567992 | 0.955175 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | edge_occlusion | -0.0418637 | -0.0751267 | -0.0086007 | t | -2.59755 | 0.0157915 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | edge_texture | 0.123312 | 0.0900494 | 0.156575 | t | 7.65127 | 6.88234e-08 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | egomotion | -0.0112323 | -0.0444953 | 0.0220307 | t | -0.696941 | 0.492536 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | fixated_pose | -0.0251512 | -0.0584142 | 0.00811175 | t | -1.56058 | 0.131713 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | inpainting | -0.0383424 | -0.0716054 | -0.00507941 | t | -2.37907 | 0.0256573 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | jigsaw | 0.00317067 | -0.0300923 | 0.0364337 | t | 0.196734 | 0.845693 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | keypoints2d | 0.0519973 | 0.0187343 | 0.0852603 | t | 3.22632 | 0.00360332 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | keypoints3d | 0.0158227 | -0.0174403 | 0.0490856 | t | 0.981762 | 0.336011 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | nonfixated_pose | -0.00315032 | -0.0364133 | 0.0301127 | t | -0.195471 | 0.84667 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | normal | -0.0142371 | -0.0475001 | 0.0190259 | t | -0.883383 | 0.385793 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | point_matching | 0.079328 | 0.0460651 | 0.112591 | t | 4.92214 | 5.06422e-05 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | random_weights | -0.0431748 | -0.0764378 | -0.0099118 | t | -2.67891 | 0.0131255 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | reshading | 0.039843 | 0.00657999 | 0.073106 | t | 2.47217 | 0.0209072 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | room_layout | -0.0601549 | -0.0934179 | -0.0268919 | t | -3.73248 | 0.00103287 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | segment_semantic | -0.00764856 | -0.0409116 | 0.0256144 | t | -0.474577 | 0.639378 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | segment_unsup25d | 0.00525102 | -0.028012 | 0.038514 | t | 0.325815 | 0.747388 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.0817987 | 0.0485358 | 0.115062 | t | 5.07544 | 3.43393e-05 | ols_fixed_effects | 50 |
| taskonomy_tasks | crsa | vanishing_point | -0.10341 | -0.136673 | -0.070147 | t | -6.41637 | 1.23521e-06 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | autoencoding | -0.237686 | -0.319366 | -0.156005 | t | -6.00581 | 3.35895e-06 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | class_object | 0.17143 | 0.0897496 | 0.253111 | t | 4.33168 | 0.000227246 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | class_scene | 0.157705 | 0.0760244 | 0.239386 | t | 3.98488 | 0.00054743 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | curvature | -0.175377 | -0.257058 | -0.0936966 | t | -4.43142 | 0.000176342 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | depth_euclidean | 0.101033 | 0.0193519 | 0.182713 | t | 2.55288 | 0.017463 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | depth_zbuffer | 0.103971 | 0.02229 | 0.185651 | t | 2.62712 | 0.0147688 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | edge_occlusion | 0.10205 | 0.0203696 | 0.183731 | t | 2.5786 | 0.0164817 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | edge_texture | -0.0203241 | -0.102005 | 0.0613565 | t | -0.513548 | 0.612263 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | egomotion | 0.023933 | -0.0577476 | 0.105614 | t | 0.604737 | 0.551029 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | fixated_pose | 0.0298547 | -0.051826 | 0.111535 | t | 0.754364 | 0.457969 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | inpainting | -0.0391978 | -0.120878 | 0.0424828 | t | -0.990446 | 0.331837 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | jigsaw | -0.0758815 | -0.157562 | 0.0057992 | t | -1.91736 | 0.0671693 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | keypoints2d | -0.107004 | -0.188684 | -0.0253231 | t | -2.70376 | 0.0123993 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | keypoints3d | 0.171521 | 0.0898403 | 0.253202 | t | 4.33398 | 0.000225926 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | nonfixated_pose | -0.0671669 | -0.148848 | 0.0145138 | t | -1.69717 | 0.102598 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | normal | 0.0959759 | 0.0142952 | 0.177657 | t | 2.42511 | 0.0231962 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | point_matching | 0.104202 | 0.0225215 | 0.185883 | t | 2.63297 | 0.0145739 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | random_weights | -0.0775898 | -0.15927 | 0.00409083 | t | -1.96053 | 0.0616404 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | reshading | 0.104366 | 0.0226855 | 0.186047 | t | 2.63711 | 0.0144374 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | room_layout | 0.0505693 | -0.0311114 | 0.13225 | t | 1.27778 | 0.213549 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | segment_semantic | 0.122296 | 0.0406152 | 0.203977 | t | 3.09016 | 0.00500424 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | segment_unsup25d | 0.121947 | 0.0402662 | 0.203628 | t | 3.08134 | 0.00511112 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.204945 | 0.123264 | 0.286626 | t | 5.17853 | 2.64611e-05 | ols_fixed_effects | 50 |
| taskonomy_tasks | wrsa | vanishing_point | -0.178258 | -0.259938 | -0.0965769 | t | -4.50419 | 0.000146537 | ols_fixed_effects | 50 |
| contrastive_self_supervised | crsa | Contrastive | 0.123065 | 0.0887984 | 0.157332 | t | 7.57718 | 7.58354e-07 | ols_fixed_effects | 20 |
| contrastive_self_supervised | wrsa | Contrastive | 0.0925768 | 0.0631422 | 0.122011 | t | 6.63573 | 4.21008e-06 | ols_fixed_effects | 20 |
| language_alignment | crsa | CLIP | -0.0726061 | -0.144156 | -0.00105626 | t | -2.17645 | 0.0471328 | ols_fixed_effects | 18 |
| language_alignment | crsa | SLIP | -0.00648079 | -0.0780307 | 0.0650691 | t | -0.194269 | 0.848756 | ols_fixed_effects | 18 |
| language_alignment | wrsa | CLIP | -0.028029 | -0.0501235 | -0.0059344 | t | -2.72086 | 0.0165656 | ols_fixed_effects | 18 |
| language_alignment | wrsa | SLIP | -0.00287544 | -0.02497 | 0.0192191 | t | -0.279127 | 0.784228 | ols_fixed_effects | 18 |
| imagenet_size | crsa | imagenet21k | 0.00846285 | -0.0038714 | 0.0207971 | z | 1.34478 | 0.178696 | mixedlm_random_intercept | 48 |
| imagenet_size | wrsa | imagenet21k | -0.0077881 | -0.0222499 | 0.00667369 | t | -1.09442 | 0.281462 | ols_fixed_effects_fallback | 48 |
| objects_faces_places | crsa | openimages | -0.0422007 | -0.0632946 | -0.0211068 | t | -6.36684 | 0.00784178 | ols_fixed_effects | 8 |
| objects_faces_places | crsa | places256 | -0.0988306 | -0.119924 | -0.0777367 | t | -14.9106 | 0.000654631 | ols_fixed_effects | 8 |
| objects_faces_places | crsa | vggface2 | 0.00011768 | -0.0209762 | 0.0212116 | t | 0.0177545 | 0.986949 | ols_fixed_effects | 8 |
| objects_faces_places | wrsa | openimages | -0.00111693 | -0.140362 | 0.138128 | t | -0.0255273 | 0.981237 | ols_fixed_effects | 8 |
| objects_faces_places | wrsa | places256 | -0.0420778 | -0.181323 | 0.0971673 | t | -0.961688 | 0.40715 | ols_fixed_effects | 8 |
| objects_faces_places | wrsa | vggface2 | -0.0560273 | -0.195272 | 0.0832178 | t | -1.2805 | 0.290395 | ols_fixed_effects | 8 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 83 | 0.4483333498239517 | 102 | 0.3049931466579437 | 83 | 19 | 15 | 0.010324346976800928 | wrsa | 117 | gmlp_s16_224_classification | 0.5827932476997375 | pure_python_grid_search_two_breakpoints | not_computed |
