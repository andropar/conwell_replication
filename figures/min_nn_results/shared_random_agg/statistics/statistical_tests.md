# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.0802112 | -0.0944683 | -0.0659541 | t | -11.0777 | 1.129e-23 | ols_fixed_effects | 270 |
| architecture | wrsa | Transformer | 0.0183693 | 0.0086694 | 0.0280691 | t | 3.72881 | 0.000235445 | ols_fixed_effects | 270 |
| taskonomy_tasks | crsa | autoencoding | -0.0682999 | -0.085187 | -0.0514128 | t | -8.02829 | 2.47525e-12 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_object | -0.00359345 | -0.0204805 | 0.0132936 | t | -0.422391 | 0.673684 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_scene | -0.0329149 | -0.0498019 | -0.0160278 | t | -3.86897 | 0.000199139 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | curvature | -0.0125672 | -0.0294543 | 0.00431985 | t | -1.47721 | 0.142893 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_euclidean | -0.0271423 | -0.0440293 | -0.0102552 | t | -3.19043 | 0.00191975 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_zbuffer | -0.00474067 | -0.0216277 | 0.0121464 | t | -0.557241 | 0.57866 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_occlusion | -0.0596105 | -0.0764976 | -0.0427235 | t | -7.0069 | 3.35862e-10 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_texture | 0.107149 | 0.0902621 | 0.124036 | t | 12.5948 | 4.73387e-22 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | egomotion | -0.0136596 | -0.0305467 | 0.00322744 | t | -1.60562 | 0.111643 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | fixated_pose | -0.0333008 | -0.0501878 | -0.0164137 | t | -3.91433 | 0.000169461 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | inpainting | -0.047081 | -0.063968 | -0.0301939 | t | -5.53412 | 2.7089e-07 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | jigsaw | 0.00801801 | -0.00886905 | 0.0249051 | t | 0.942475 | 0.348315 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints2d | 0.0387993 | 0.0219122 | 0.0556863 | t | 4.56065 | 1.5056e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints3d | 0.012289 | -0.00459809 | 0.029176 | t | 1.4445 | 0.151854 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | nonfixated_pose | 0.00040926 | -0.0164778 | 0.0172963 | t | 0.0481063 | 0.961731 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | normal | -0.0307569 | -0.047644 | -0.0138698 | t | -3.61531 | 0.000480195 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | point_matching | 0.0644885 | 0.0476014 | 0.0813755 | t | 7.58028 | 2.17291e-11 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | random_weights | -0.0537971 | -0.0706842 | -0.0369101 | t | -6.32357 | 8.06235e-09 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | reshading | 0.0240177 | 0.00713067 | 0.0409048 | t | 2.82316 | 0.00578176 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | room_layout | -0.067686 | -0.084573 | -0.0507989 | t | -7.95612 | 3.51778e-12 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_semantic | -0.0230365 | -0.0399236 | -0.00614946 | t | -2.70782 | 0.00801871 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup25d | -0.00180767 | -0.0186947 | 0.0150794 | t | -0.212482 | 0.832182 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.0735512 | 0.0566641 | 0.0904382 | t | 8.64555 | 1.20167e-13 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | vanishing_point | -0.100814 | -0.117701 | -0.0839267 | t | -11.8501 | 1.70415e-20 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | autoencoding | -0.248004 | -0.298112 | -0.197896 | t | -9.82444 | 3.54418e-16 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_object | 0.119686 | 0.069578 | 0.169794 | t | 4.74125 | 7.37081e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_scene | 0.114203 | 0.0640944 | 0.164311 | t | 4.52402 | 1.73695e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | curvature | -0.194609 | -0.244717 | -0.144501 | t | -7.70926 | 1.16555e-11 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_euclidean | 0.0595492 | 0.00944106 | 0.109657 | t | 2.35898 | 0.0203511 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_zbuffer | 0.0612936 | 0.0111855 | 0.111402 | t | 2.42809 | 0.0170415 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_occlusion | 0.0521163 | 0.00200824 | 0.102224 | t | 2.06454 | 0.0416635 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_texture | -0.0162315 | -0.0663396 | 0.0338766 | t | -0.642996 | 0.52176 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | egomotion | -0.0177639 | -0.067872 | 0.0323442 | t | -0.703699 | 0.483323 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | fixated_pose | -0.00078346 | -0.0508916 | 0.0493246 | t | -0.031036 | 0.975305 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | inpainting | -0.0797089 | -0.129817 | -0.0296008 | t | -3.15759 | 0.00212648 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | jigsaw | -0.108733 | -0.158842 | -0.0586254 | t | -4.30737 | 3.99082e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints2d | -0.0989098 | -0.149018 | -0.0488017 | t | -3.91822 | 0.000167126 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints3d | 0.131781 | 0.0816728 | 0.181889 | t | 5.22037 | 1.03228e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | nonfixated_pose | -0.0718589 | -0.121967 | -0.0217508 | t | -2.84662 | 0.0054034 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | normal | 0.0562043 | 0.00609618 | 0.106312 | t | 2.22648 | 0.0283204 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | point_matching | 0.0443431 | -0.00576504 | 0.0944512 | t | 1.75661 | 0.0821734 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | random_weights | -0.129134 | -0.179242 | -0.0790257 | t | -5.11551 | 1.60054e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | reshading | 0.0802124 | 0.0301043 | 0.130321 | t | 3.17754 | 0.00199856 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | room_layout | 0.0179034 | -0.0322047 | 0.0680115 | t | 0.709224 | 0.479904 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_semantic | 0.0650155 | 0.0149074 | 0.115124 | t | 2.57553 | 0.0115343 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup25d | 0.0552137 | 0.00510561 | 0.105322 | t | 2.18724 | 0.0311537 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.117577 | 0.0674685 | 0.167685 | t | 4.65768 | 1.02773e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | vanishing_point | -0.188679 | -0.238788 | -0.138571 | t | -7.47435 | 3.61769e-11 | ols_fixed_effects | 125 |
| contrastive_self_supervised | crsa | Contrastive | 0.110482 | 0.0901701 | 0.130795 | t | 10.962 | 3.64626e-14 | ols_fixed_effects | 50 |
| contrastive_self_supervised | wrsa | Contrastive | 0.0791491 | 0.0626112 | 0.0956871 | t | 9.64538 | 2.01031e-12 | ols_fixed_effects | 50 |
| language_alignment | crsa | CLIP | -0.0497017 | -0.0894855 | -0.00991803 | t | -2.52907 | 0.0157075 | ols_fixed_effects | 45 |
| language_alignment | crsa | SLIP | 0.00105775 | -0.038726 | 0.0408415 | t | 0.0538236 | 0.957358 | ols_fixed_effects | 45 |
| language_alignment | wrsa | CLIP | -0.017566 | -0.0323212 | -0.00281082 | t | -2.41004 | 0.0208991 | ols_fixed_effects | 45 |
| language_alignment | wrsa | SLIP | 0.00800591 | -0.00674927 | 0.0227611 | t | 1.0984 | 0.27894 | ols_fixed_effects | 45 |
| imagenet_size | crsa | imagenet21k | 0.0113923 | 0.00319485 | 0.0195898 | z | 2.72383 | 0.00645293 | mixedlm_random_intercept | 120 |
| imagenet_size | wrsa | imagenet21k | -0.0119741 | -0.0201585 | -0.00378966 | z | -2.86749 | 0.00413741 | mixedlm_random_intercept | 120 |
| objects_faces_places | crsa | openimages | -0.0366271 | -0.0521387 | -0.0211156 | t | -5.14479 | 0.000243071 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | places256 | -0.0946749 | -0.110186 | -0.0791634 | t | -13.2984 | 1.52722e-08 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | vggface2 | 0.0161327 | 0.000621141 | 0.0316442 | t | 2.26606 | 0.0427409 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | openimages | -0.0058972 | -0.0669571 | 0.0551627 | t | -0.210431 | 0.836862 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | places256 | -0.0468616 | -0.107921 | 0.0141983 | t | -1.67217 | 0.12034 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | vggface2 | -0.0576912 | -0.118751 | 0.0033687 | t | -2.05861 | 0.0619232 | ols_fixed_effects | 20 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 84 | 0.43257714033126826 | 107 | 0.20492634415626526 | 84 | 23 | 10 | 0.008474150961243295 | wrsa | 117 | levit_128_classification | 0.5465446841716767 | pure_python_grid_search_two_breakpoints | not_computed |
