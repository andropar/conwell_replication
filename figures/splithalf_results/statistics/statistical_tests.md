# Conwell-Style Statistical Tests

Model-to-model RSA is not included here; it requires saved best-layer RSM vectors.

## Targeted Comparisons

| comparison_id | eval_type | level | beta | ci_lo | ci_hi | statistic_name | statistic | p_value | model_type | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | crsa | Transformer | -0.080403 | -0.0929516 | -0.0678543 | t | -12.6148 | 5.66409e-29 | ols_fixed_effects | 275 |
| architecture | wrsa | Transformer | 0.0114205 | -0.00483696 | 0.027678 | t | 1.3833 | 0.167766 | ols_fixed_effects | 265 |
| taskonomy_tasks | crsa | autoencoding | -0.0272732 | -0.0658453 | 0.011299 | t | -1.40352 | 0.163688 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_object | -0.039666 | -0.0782382 | -0.00109388 | t | -2.04128 | 0.0439676 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | class_scene | -0.0116936 | -0.0502658 | 0.0268785 | t | -0.601772 | 0.548744 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | curvature | -0.036373 | -0.0749452 | 0.00219913 | t | -1.87181 | 0.0642769 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_euclidean | -0.0178347 | -0.0564069 | 0.0207374 | t | -0.917804 | 0.361022 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | depth_zbuffer | -0.0350274 | -0.0735995 | 0.00354479 | t | -1.80256 | 0.0745947 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_occlusion | -0.00245827 | -0.0410304 | 0.0361139 | t | -0.126506 | 0.899596 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | edge_texture | 0.0247099 | -0.0138623 | 0.063282 | t | 1.27161 | 0.206584 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | egomotion | -0.0333972 | -0.0719693 | 0.005175 | t | -1.71867 | 0.0888977 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | fixated_pose | -0.0646635 | -0.103236 | -0.0260913 | t | -3.32769 | 0.00124246 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | inpainting | -0.0752366 | -0.113809 | -0.0366645 | t | -3.8718 | 0.000197152 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | jigsaw | 0.0345292 | -0.00404291 | 0.0731014 | t | 1.77693 | 0.0787472 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints2d | -0.0667742 | -0.105346 | -0.028202 | t | -3.43631 | 0.000872998 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | keypoints3d | 0.0179678 | -0.0206043 | 0.05654 | t | 0.924653 | 0.357465 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | nonfixated_pose | 0.014963 | -0.0236091 | 0.0535352 | t | 0.77002 | 0.443179 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | normal | -0.00671732 | -0.0452895 | 0.0318548 | t | -0.345684 | 0.730337 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | point_matching | 0.0298658 | -0.00870636 | 0.0684379 | t | 1.53694 | 0.127597 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | random_weights | -0.0530064 | -0.0915786 | -0.0144342 | t | -2.72779 | 0.00758225 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | reshading | -0.00739062 | -0.0459628 | 0.0311815 | t | -0.380333 | 0.704538 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | room_layout | -0.0557324 | -0.0943045 | -0.0171602 | t | -2.86808 | 0.00507745 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_semantic | -0.00106849 | -0.0396406 | 0.0375037 | t | -0.0549862 | 0.956264 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup25d | 0.0215484 | -0.0170237 | 0.0601206 | t | 1.10892 | 0.270237 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | segment_unsup2d | 0.0467405 | 0.00816837 | 0.0853127 | t | 2.40534 | 0.0180738 | ols_fixed_effects | 125 |
| taskonomy_tasks | crsa | vanishing_point | -0.125568 | -0.16414 | -0.0869955 | t | -6.46191 | 4.27446e-09 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | autoencoding | -0.25905 | -0.297133 | -0.220967 | t | -13.5023 | 6.46687e-24 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_object | 0.0712575 | 0.0331744 | 0.109341 | t | 3.71411 | 0.000342372 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | class_scene | 0.065289 | 0.0272059 | 0.103372 | t | 3.40302 | 0.000973486 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | curvature | -0.197301 | -0.235385 | -0.159218 | t | -10.2838 | 3.65822e-17 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_euclidean | 0.0411094 | 0.00302619 | 0.0791925 | t | 2.14272 | 0.0346647 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | depth_zbuffer | 0.0233173 | -0.0147659 | 0.0614004 | t | 1.21535 | 0.227213 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_occlusion | 0.0335303 | -0.0045529 | 0.0716134 | t | 1.74768 | 0.0837175 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | edge_texture | -0.0432312 | -0.0813144 | -0.00514808 | t | -2.25331 | 0.026515 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | egomotion | -0.077225 | -0.115308 | -0.0391418 | t | -4.02515 | 0.000113693 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | fixated_pose | -0.0264746 | -0.0645577 | 0.0116086 | t | -1.37992 | 0.170818 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | inpainting | -0.0410986 | -0.0791818 | -0.00301548 | t | -2.14216 | 0.0347108 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | jigsaw | -0.132786 | -0.170869 | -0.0947029 | t | -6.92112 | 5.03357e-10 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints2d | -0.149425 | -0.187508 | -0.111342 | t | -7.78837 | 7.9459e-12 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | keypoints3d | 0.0889318 | 0.0508486 | 0.127015 | t | 4.63533 | 1.12267e-05 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | nonfixated_pose | -0.133563 | -0.171647 | -0.0954803 | t | -6.96164 | 4.15867e-10 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | normal | -0.0479938 | -0.0860769 | -0.00991061 | t | -2.50155 | 0.0140574 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | point_matching | -0.0260552 | -0.0641384 | 0.012028 | t | -1.35806 | 0.177628 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | random_weights | -0.171569 | -0.209653 | -0.133486 | t | -8.9426 | 2.7779e-14 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | reshading | 0.0622199 | 0.0241367 | 0.100303 | t | 3.24305 | 0.00162718 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | room_layout | 0.00345039 | -0.0346328 | 0.0415336 | t | 0.179843 | 0.857655 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_semantic | 0.0230056 | -0.0150775 | 0.0610888 | t | 1.19911 | 0.233438 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup25d | 0.0249714 | -0.0131118 | 0.0630546 | t | 1.30157 | 0.196179 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | segment_unsup2d | 0.0973268 | 0.0592436 | 0.13541 | t | 5.0729 | 1.91029e-06 | ols_fixed_effects | 125 |
| taskonomy_tasks | wrsa | vanishing_point | -0.138799 | -0.176883 | -0.100716 | t | -7.23455 | 1.13999e-10 | ols_fixed_effects | 125 |
| contrastive_self_supervised | crsa | Contrastive | 0.0655331 | 0.0404113 | 0.0906549 | t | 5.25731 | 4.10425e-06 | ols_fixed_effects | 50 |
| contrastive_self_supervised | wrsa | Contrastive | 0.111315 | 0.0849544 | 0.137676 | t | 8.5104 | 7.55161e-11 | ols_fixed_effects | 50 |
| language_alignment | crsa | CLIP | 0.00424249 | -0.0175028 | 0.0259878 | t | 0.394957 | 0.695083 | ols_fixed_effects | 45 |
| language_alignment | crsa | SLIP | -0.00983687 | -0.0315822 | 0.0119085 | t | -0.915769 | 0.365563 | ols_fixed_effects | 45 |
| language_alignment | wrsa | CLIP | -0.0268981 | -0.0472863 | -0.00650998 | t | -2.67079 | 0.011077 | ols_fixed_effects | 45 |
| language_alignment | wrsa | SLIP | 0.0254516 | 0.00506346 | 0.0458398 | t | 2.52716 | 0.0157807 | ols_fixed_effects | 45 |
| imagenet_size | crsa | imagenet21k | 0.0106331 | 0.000312025 | 0.0209541 | z | 2.01922 | 0.0434646 | mixedlm_random_intercept | 120 |
| imagenet_size | wrsa | imagenet21k | 0.00409932 | -0.00758844 | 0.0157871 | z | 0.68743 | 0.491812 | mixedlm_random_intercept | 120 |
| objects_faces_places | crsa | openimages | -0.0243098 | -0.0447776 | -0.003842 | t | -2.5878 | 0.0237541 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | places256 | -0.0768111 | -0.0972789 | -0.0563433 | t | -8.1766 | 3.0046e-06 | ols_fixed_effects | 20 |
| objects_faces_places | crsa | vggface2 | -0.00342742 | -0.0238952 | 0.0170404 | t | -0.364852 | 0.721571 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | openimages | -0.027304 | -0.0963356 | 0.0417276 | t | -0.861783 | 0.405699 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | places256 | -0.102274 | -0.171306 | -0.0332424 | t | -3.22803 | 0.00724575 | ols_fixed_effects | 20 |
| objects_faces_places | wrsa | vggface2 | -0.0995069 | -0.168538 | -0.0304753 | t | -3.14069 | 0.00852098 | ols_fixed_effects | 20 |

## Breakpoint

| breakpoint_1_rank | breakpoint_1_score | breakpoint_2_rank | breakpoint_2_score | n_models_top_segment | n_models_middle_segment | n_models_bottom_segment | sse | metric | n_models | top_model | top_score | method | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 78 | 0.4074413239955902 | 106 | 0.1430153653025627 | 78 | 28 | 10 | 0.008867916856085383 | wrsa | 116 | gmlp_s16_224_classification | 0.5426120162010193 | pure_python_grid_search_two_breakpoints | not_computed |
