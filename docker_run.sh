#!/bin/bash
set -e

username=yyang
project_name=phys-gs
container_name=phys_gs
folder_name=Phys-GS
workspace_root=/mnt/workspace2026/${username}

xhost +local:root
xhost +si:localuser:${username}

docker run --gpus all -itd \
    -u $(id -u $username):$(id -g $username) \
    --name ${username}_${container_name} \
    -v ${workspace_root}/${folder_name}:/home/${username}/mnt/workspace \
    --mount type=bind,source="/mnt/poplin/share/2023/users/yang/",target=/mnt/data \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=${XAUTHORITY:-/home/${username}/.Xauthority} \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${XAUTHORITY:-/home/${username}/.Xauthority}:${XAUTHORITY:-/home/${username}/.Xauthority} \
    repo-luna.ist.osaka-u.ac.jp:5000/${username}/${project_name}:build
