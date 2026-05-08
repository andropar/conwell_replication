# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.056939 | -0.0684164 | -0.0454616 | t | -9.76811 | 1.93722e-19 | ols_fixed_effects | 270 |
| architecture | wrsa | Transformer | 0.0135509 | 0.00554992 | 0.0215519 | t | 3.33479 | 0.000976071 | ols_fixed_effects | 270 |
| taskonomy_tasks | crsa | autoencoding | -0.0505408 | -0.0642973 | -0.0367842 | t | -7.29273 | 8.63645e-11 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_object | 0.032588 | 0.0188314 | 0.0463445 | t | 4.70225 | 8.61134e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_scene | 0.0142715 | 0.000514986 | 0.028028 | t | 2.05929 | 0.0421737 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | curvature | 0.0140464 | 0.000289887 | 0.0278029 | t | 2.02681 | 0.0454548 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_euclidean | 0.0165673 | 0.00281073 | 0.0303238 | t | 2.39056 | 0.0187742 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_zbuffer | 0.0343713 | 0.0206148 | 0.0481278 | t | 4.95958 | 3.04668e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_occlusion | -0.0178159 | -0.0315724 | -0.00405933 | t | -2.57072 | 0.0116849 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_texture | 0.0969072 | 0.0831506 | 0.110664 | t | 13.9831 | 6.8966e-25 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | egomotion | 0.00180331 | -0.0119532 | 0.0155598 | t | 0.260206 | 0.795262 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | fixated_pose | -0.00410755 | -0.0178641 | 0.00964897 | t | -0.592695 | 0.554779 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | inpainting | -0.0408126 | -0.0545691 | -0.027056 | t | -5.88901 | 5.71652e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | jigsaw | 0.0352387 | 0.0214821 | 0.0489952 | t | 5.08473 | 1.81887e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints2d | 0.0213147 | 0.00755823 | 0.0350713 | t | 3.07559 | 0.0027366 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints3d | 0.0464778 | 0.0327213 | 0.0602343 | t | 6.70647 | 1.37624e-09 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | nonfixated_pose | 0.00951734 | -0.00423918 | 0.0232739 | t | 1.3733 | 0.172859 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | normal | 0.0238487 | 0.0100922 | 0.0376052 | t | 3.44122 | 0.000859008 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | point_matching | 0.0520609 | 0.0383043 | 0.0658174 | t | 7.51207 | 3.01767e-11 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | random_weights | -0.0324436 | -0.0462001 | -0.0186871 | t | -4.68142 | 9.35447e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | reshading | 0.0403276 | 0.0265711 | 0.0540841 | t | 5.81903 | 7.79476e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | room_layout | -0.0226278 | -0.0363843 | -0.00887125 | t | -3.26505 | 0.00151765 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_semantic | 0.0246032 | 0.0108467 | 0.0383597 | t | 3.5501 | 0.000598396 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup25d | 0.0332386 | 0.019482 | 0.0469951 | t | 4.79613 | 5.91499e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.0713519 | 0.0575954 | 0.0851085 | t | 10.2957 | 3.45026e-17 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | vanishing_point | -0.0663365 | -0.0800931 | -0.05258 | t | -9.57197 | 1.23628e-15 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | autoencoding | -0.170365 | -0.207728 | -0.133003 | t | -9.05106 | 1.62585e-14 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_object | 0.147236 | 0.109873 | 0.184599 | t | 7.82224 | 6.74236e-12 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_scene | 0.158889 | 0.121526 | 0.196252 | t | 8.44135 | 3.27958e-13 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | curvature | -0.119744 | -0.157106 | -0.0823809 | t | -6.36166 | 6.77323e-09 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_euclidean | 0.108159 | 0.0707967 | 0.145522 | t | 5.74622 | 1.07445e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_zbuffer | 0.10579 | 0.0684273 | 0.143153 | t | 5.62035 | 1.86356e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_occlusion | 0.102001 | 0.0646382 | 0.139364 | t | 5.41904 | 4.44404e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_texture | 0.0241813 | -0.0131815 | 0.0615441 | t | 1.28469 | 0.201993 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | egomotion | 0.0554362 | 0.0180735 | 0.092799 | t | 2.94518 | 0.00404949 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | fixated_pose | 0.0423165 | 0.00495373 | 0.0796793 | t | 2.24816 | 0.0268536 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | inpainting | 0.0080437 | -0.0293191 | 0.0454065 | t | 0.42734 | 0.670089 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | jigsaw | -0.0060818 | -0.0434446 | 0.031281 | t | -0.32311 | 0.747315 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints2d | -0.0409069 | -0.0782697 | -0.00354415 | t | -2.17328 | 0.0322199 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints3d | 0.128219 | 0.0908559 | 0.165581 | t | 6.81192 | 8.40702e-10 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | nonfixated_pose | 0.00983259 | -0.0275302 | 0.0471954 | t | 0.522379 | 0.60261 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | normal | 0.0479342 | 0.0105715 | 0.085297 | t | 2.54662 | 0.0124673 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | point_matching | 0.0886303 | 0.0512675 | 0.125993 | t | 4.70869 | 8.39335e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | random_weights | -0.0627561 | -0.100119 | -0.0253933 | t | -3.33406 | 0.00121723 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | reshading | 0.10169 | 0.0643277 | 0.139053 | t | 5.40254 | 4.76886e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | room_layout | 0.0775145 | 0.0401517 | 0.114877 | t | 4.11814 | 8.09064e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_semantic | 0.111933 | 0.0745701 | 0.149296 | t | 5.9467 | 4.4219e-08 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup25d | 0.0983362 | 0.0609734 | 0.135699 | t | 5.22434 | 1.01521e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.0901863 | 0.0528236 | 0.127549 | t | 4.79136 | 6.0295e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | vanishing_point | -0.130353 | -0.167716 | -0.0929902 | t | -6.92531 | 4.93538e-10 | ols_fixed_effects | 125 |
| contrastive_self_supervised | crsa | Contrastive | 0.0593598 | 0.04104 | 0.0776796 | t | 6.53018 | 5.60035e-08 | ols_fixed_effects | 50 |
| contrastive_self_supervised | wrsa | Contrastive | 0.050467 | 0.036298 | 0.064636 | t | 7.17831 | 6.26794e-09 | ols_fixed_effects | 50 |
| language_alignment | crsa | CLIP | -0.05097 | -0.0782567 | -0.0236833 | t | -3.78146 | 0.000536947 | ols_fixed_effects | 45 |
| language_alignment | crsa | SLIP | 0.0116555 | -0.0156311 | 0.0389422 | t | 0.864722 | 0.392618 | ols_fixed_effects | 45 |
| language_alignment | wrsa | CLIP | -0.0202709 | -0.0306978 | -0.00984396 | t | -3.9356 | 0.00034168 | ols_fixed_effects | 45 |
| language_alignment | wrsa | SLIP | 0.0073176 | -0.00310934 | 0.0177445 | t | 1.42071 | 0.163554 | ols_fixed_effects | 45 |
| imagenet_size | crsa | imagenet21k | 0.0118957 | 0.00446718 | 0.0193242 | z | 3.1386 | 0.00169758 | mixedlm_random_intercept | 120 |
| imagenet_size | wrsa | imagenet21k | -0.00933909 | -0.014743 | -0.00393515 | z | -3.38721 | 0.000706066 | mixedlm_random_intercept | 120 |
| objects_faces_places | crsa | openimages | -0.0229232 | -0.031652 | -0.0141943 | t | -5.72186 | 9.57767e-05 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | places256 | -0.0768305 | -0.0855594 | -0.0681016 | t | -19.1777 | 2.27587e-10 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | vggface2 | 0.000868203 | -0.00786066 | 0.00959706 | t | 0.216712 | 0.832073 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | openimages | -0.017765 | -0.0458925 | 0.0103625 | t | -1.37611 | 0.193925 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | places256 | -0.058836 | -0.0869635 | -0.0307086 | t | -4.55756 | 0.0006575 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | vggface2 | -0.11851 | -0.146638 | -0.0903829 | t | -9.18006 | 8.95247e-07 | ols_fixed_effects | 20 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 91 | 0.40092491507530204 | 107 | 0.211155177950859 | 91 | 16 | 10 | 0.010369605815986416 | wrsa | 117 | swin_tiny_patch4_window7_224_classification | 0.5287981009483337 | pure_python_grid_search_two_breakpoints | not_computed |
