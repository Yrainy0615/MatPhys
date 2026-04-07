#!/bin/bash
set -e

username=yyang
project_name=phys-gs
container_name=phys_gs
folder_name=Phys-GS
workspace_root=/mnt/workspace2026/${username}

docker run --gpus all -itd \
    -u $(id -u $username):$(id -g $username) \
    --name ${username}_${container_name} \
    -v ${workspace_root}/${folder_name}:/home/${username}/mnt/workspace \
    --mount type=bind,source="/mnt/poplin/share/2023/users/yang/",target=/mnt/data \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    repo-luna.ist.osaka-u.ac.jp:5000/${username}/${project_name}:build
