import os

import open3d as o3d
import numpy as np
import torch
import time
import cv2
from .config import cfg


def _open_video_writer(save_path, fps, width, height):
    """Open a VideoWriter with codec fallback and explicit failure reporting."""
    ext = os.path.splitext(save_path)[1].lower()
    codec_candidates = ["mp4v", "avc1"] if ext == ".mp4" else ["MJPG", "mp4v", "avc1"]

    last_error = None
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
        last_error = codec

    raise RuntimeError(
        f"Failed to initialize VideoWriter for {save_path} "
        f"(tried codecs: {', '.join(codec_candidates)}; last={last_error})"
    )


def _render_frame_projected(
    points,
    colors,
    intrinsic,
    w2c,
    width,
    height,
    overlay_path=None,
    frame_idx=None,
    point_radius=2,
):
    """Headless fallback renderer: project points and splat colored circles."""
    # Auto-scale Cupid-style normalized intrinsics ([0,1] image plane) to pixels.
    intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
    if intrinsic.shape == (3, 3) and float(np.max(np.abs(intrinsic[:2, :]))) <= 2.0:
        intrinsic[0, :] *= float(width)
        intrinsic[1, :] *= float(height)
    if overlay_path is not None and frame_idx is not None:
        image_path = f"{overlay_path}/{frame_idx}.png"
        overlay = cv2.imread(image_path)
        if overlay is not None:
            frame = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        else:
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    else:
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([points.astype(np.float32), ones], axis=1)
    cam = pts_h @ w2c.T
    z = cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return frame

    cam = cam[valid]
    draw_colors = colors[valid]
    z = z[valid]

    u = intrinsic[0, 0] * (cam[:, 0] / z) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (cam[:, 1] / z) + intrinsic[1, 2]

    order = np.argsort(z)[::-1]
    u = u[order]
    v = v[order]
    draw_colors = draw_colors[order]

    for px, py, color in zip(u, v, draw_colors):
        x = int(round(px))
        y = int(round(py))
        if 0 <= x < width and 0 <= y < height:
            rgb = np.clip(color * 255.0, 0, 255).astype(np.uint8).tolist()
            cv2.circle(frame, (x, y), point_radius, rgb, thickness=-1, lineType=cv2.LINE_AA)

    return frame


def visualize_pc(
    object_points,
    object_colors=None,
    controller_points=None,
    object_visibilities=None,
    object_motions_valid=None,
    visualize=True,
    save_video=False,
    save_path=None,
    vis_cam_idx=0,
    width=None,
    height=None,
    intrinsic=None,
    w2c=None,
    overlay_path=None,
):
    # Deprecated function, use visualize_pc instead
    FPS = cfg.FPS
    if width is None or height is None:
        width, height = cfg.WH
    if intrinsic is None:
        intrinsic = cfg.intrinsics[vis_cam_idx]
    if w2c is None:
        w2c = cfg.w2cs[vis_cam_idx]
    overlay_root = cfg.overlay_path if overlay_path is None else overlay_path

    # Convert the stuffs to numpy if it's tensor
    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(object_colors, torch.Tensor):
        object_colors = object_colors.cpu().numpy()
    if isinstance(object_visibilities, torch.Tensor):
        object_visibilities = object_visibilities.cpu().numpy()
    if isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = object_motions_valid.cpu().numpy()
    if isinstance(controller_points, torch.Tensor):
        controller_points = controller_points.cpu().numpy()

    if object_colors is None:
        object_colors = np.tile(
            [1, 0, 0], (object_points.shape[0], object_points.shape[1], 1)
        )
    else:
        if object_colors.shape[1] < object_points.shape[1]:
            # If the object_colors is not the same as object_points, fill the colors with black
            object_colors = np.concatenate(
                [
                    object_colors,
                    np.ones(
                        (
                            object_colors.shape[0],
                            object_points.shape[1] - object_colors.shape[1],
                            3,
                        )
                    )
                    * 0.3,
                ],
                axis=1,
            )

    # The pcs is a 4d pcd numpy array with shape (n_frames, n_points, 3)
    vis = None
    use_headless_fallback = False
    if visualize or save_video:
        headless_env = not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
        if headless_env and save_video and not visualize:
            use_headless_fallback = True
        else:
            vis = o3d.visualization.Visualizer()
            created = vis.create_window(visible=visualize, width=width, height=height)
            if not created:
                use_headless_fallback = True

    if save_video and visualize:
        raise ValueError("Cannot save video and visualize at the same time.")

    # Initialize video writer if save_video is True
    if save_video:
        video_writer = _open_video_writer(save_path, FPS, width, height)

    if controller_points is not None and not use_headless_fallback:
        controller_meshes = []
        prev_center = []
    for i in range(object_points.shape[0]):
        object_pcd = o3d.geometry.PointCloud()
        if object_visibilities is None:
            object_pcd.points = o3d.utility.Vector3dVector(object_points[i])
            object_pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
        else:
            object_pcd.points = o3d.utility.Vector3dVector(
                object_points[i, np.where(object_visibilities[i])[0], :]
            )
            object_pcd.colors = o3d.utility.Vector3dVector(
                object_colors[i, np.where(object_visibilities[i])[0], :]
            )
        if use_headless_fallback:
            if object_visibilities is None:
                frame_points = object_points[i]
                frame_colors = object_colors[i]
            else:
                valid_idx = np.where(object_visibilities[i])[0]
                frame_points = object_points[i, valid_idx, :]
                frame_colors = object_colors[i, valid_idx, :]
            frame = _render_frame_projected(
                frame_points,
                frame_colors,
                intrinsic,
                w2c,
                width,
                height,
                overlay_path=f"{overlay_root}/{vis_cam_idx}" if overlay_root is not None else None,
                frame_idx=i,
            )
            if save_video:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if visualize:
                cv2.imshow("visualize_pc", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(max(1, int(1000 / FPS)))
            continue

        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            if controller_points is not None:
                # Use sphere mesh for each controller point
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    origin_color = [1, 0, 0]
                    controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.01
                    ).translate(origin)
                    controller_mesh.compute_vertex_normals()
                    controller_mesh.paint_uniform_color(origin_color)
                    controller_meshes.append(controller_mesh)
                    vis.add_geometry(controller_meshes[-1])
                    prev_center.append(origin)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                width, height, intrinsic
            )
            camera_params.intrinsic = intrinsic_parameter
            camera_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(
                camera_params, allow_arbitrary=True
            )
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            if controller_points is not None:
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    controller_meshes[j].translate(origin - prev_center[j])
                    vis.update_geometry(controller_meshes[j])
                    prev_center[j] = origin
        vis.poll_events()
        vis.update_renderer()

        # Capture frame and write to video file if save_video is True
        if save_video:
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            if overlay_root is not None:
                # Get the mask where the pixel is white
                mask = np.all(frame == [255, 255, 255], axis=-1)
                image_path = f"{overlay_root}/{vis_cam_idx}/{i}.png"
                overlay = cv2.imread(image_path)
                if overlay is not None:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    frame[mask] = overlay[mask]
            # Convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        if visualize:
            time.sleep(1 / FPS)

    if vis is not None:
        vis.destroy_window()
    if use_headless_fallback and visualize:
        cv2.destroyAllWindows()
    if save_video:
        video_writer.release()
