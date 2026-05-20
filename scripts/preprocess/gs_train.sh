#!/bin/bash
cd "$(dirname "$0")/../.."
set -e
output_dir="./gaussian_output"
output_video_dir="./gaussian_output_video"
# scenes=("double_lift_cloth_1" "double_lift_cloth_3" "double_lift_sloth" "double_lift_zebra"
#         "double_stretch_sloth" "double_stretch_zebra"
#         "rope_double_hand"
#         "single_clift_cloth_1" "single_clift_cloth_3"
#         "single_lift_cloth" "single_lift_cloth_1" "single_lift_cloth_3" "single_lift_cloth_4"
#         "single_lift_dinosor" "single_lift_rope" "single_lift_sloth" "single_lift_zebra"
#         "single_push_rope" "single_push_rope_1" "single_push_rope_4"
#         "single_push_sloth"
#         "weird_package")

scenes=("double_stretch_sloth")

exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

python ./gaussian_splatting/generate_interp_poses.py

# Iterate over each folder
for scene_name in "${scenes[@]}"; do
    echo "Processing: $scene_name"

    # Training
    python gs_train.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ${output_dir}/${scene_name}/${exp_name} \
        --iterations 10000 \
        --lambda_depth 0.001 \
        --lambda_normal 0.0 \
        --lambda_anisotropic 0.0 \
        --lambda_seg 1.0 \
        --use_masks \
        --isotropic \
        --gs_init_opt 'hybrid'

    # Rendering
    python gs_render.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ${output_dir}/${scene_name}/${exp_name} \

    # Convert images to video
    python gaussian_splatting/img2video.py \
        --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
        --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
done
