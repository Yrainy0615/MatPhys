"""
Lift 2D Co-Tracker tracks to 3D using a Cupid mesh.

Strategy (per user choice):
    1) For frame 0, ray-cast each query pixel against the Cupid mesh in model
       space, recording the hit point and its camera-frame depth z0.
    2) For frames t>0, re-use z0 and back-project the new 2D pixel
       (from cotracker) to 3D in model space.

The Cupid intrinsic is in normalized image coordinates (image plane [0,1]^2
after pad_to_square). We map an original-frame pixel (px, py) to the padded
square via S = max(W, H); pad offsets ((S-W)/2, (S-H)/2); normalize by S.

Inputs:
    <case>/color/0.mp4                  (for HxW)
    <case>/cotracker/0.npz              tracks (T,N,2) (y,x), visibility (T,N)
    <case>/mask/0/<obj_id>/0.png        per-object first-frame masks
    <case>/mask/mask_info_0.json        obj_id -> name
    results/<case>/cupid/mesh0.glb      Cupid mesh in model frame
    results/<case>/cupid/pose.json      extrinsic (model->cam), intrinsic

Outputs (written under <case>):
    object_tracks_3d.npz
        points       : float32 [T, N_obj, 3]   in model (=world) frame
        visibility   : bool    [T, N_obj]
        colors       : float32 [N_obj, 3]      sampled from frame 0
        valid_first  : bool    [N_obj]         (True where ray hit mesh)
    controller_tracks_3d.npz   (only if controller mask exists)
        points       : float32 [T, N_ctrl, 3]
        visibility   : bool    [T, N_ctrl]
"""

import argparse
import glob
import json
import os

import cv2
import imageio.v3 as iio
import numpy as np
import trimesh


CONTROLLER_NAME = "hand"


def load_mask(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 0


def find_mesh(cupid_dir):
    candidates = sorted(glob.glob(os.path.join(cupid_dir, "mesh*.glb")))
    if not candidates:
        raise FileNotFoundError(f"no mesh*.glb in {cupid_dir}")
    return candidates[0]


def load_trimesh(path):
    scene_or_mesh = trimesh.load(path, force="mesh")
    if isinstance(scene_or_mesh, trimesh.Scene):
        scene_or_mesh = trimesh.util.concatenate(
            [g for g in scene_or_mesh.geometry.values()]
        )
    if not isinstance(scene_or_mesh, trimesh.Trimesh):
        raise RuntimeError(f"loaded mesh has unexpected type: {type(scene_or_mesh)}")
    return scene_or_mesh


def build_rays_cam(px, py, W, H, intrinsic):
    """Return [N, 3] ray directions in camera frame for original-frame pixels."""
    S = max(W, H)
    u = (px + (S - W) / 2.0) / S
    v = (py + (S - H) / 2.0) / S
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    dz = np.ones_like(dx)
    return np.stack([dx, dy, dz], axis=1)


def cast_first_frame(mesh, ray_origins_world, ray_dirs_world):
    """Return hit points (model frame) and a valid mask."""
    locations, ray_indices, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins_world,
        ray_directions=ray_dirs_world,
        multiple_hits=False,
    )
    N = ray_origins_world.shape[0]
    hits = np.full((N, 3), np.nan, dtype=np.float64)
    valid = np.zeros(N, dtype=bool)
    if locations.shape[0] > 0:
        hits[ray_indices] = locations
        valid[ray_indices] = True
    return hits, valid


def _compute_motions_valid_neighbor(
    pts: np.ndarray, vis: np.ndarray,
    neighbor_dist: float = 0.03,
    min_neighbors: int = 5,
) -> np.ndarray:
    """Reproduce data_process_track.py motion-consistency filter for single-view.

    pts: (T, N, 3) world-frame tracks (may contain NaN).
    vis: (T, N) bool visibility.
    Returns motions_valid (T, N) bool where motions_valid[t, j] = True iff:
      - vis[t, j] and vis[t+1, j], and
      - at least min_neighbors points within neighbor_dist of pts[t, j] are valid, and
      - at least half of them have motion close to pts[t, j]'s motion.
    """
    from scipy.spatial import cKDTree
    T, N, _ = pts.shape
    motions = np.zeros_like(pts)
    motions[:-1] = pts[1:] - pts[:-1]
    mv = np.zeros_like(vis)
    mv[:-1] = vis[:-1] & vis[1:]
    half = neighbor_dist * 0.5
    for t in range(T - 1):
        valid_mask = mv[t] & np.isfinite(pts[t]).all(-1)
        if valid_mask.sum() < min_neighbors:
            mv[t] = False
            continue
        valid_ids = np.where(valid_mask)[0]
        tree = cKDTree(pts[t, valid_ids])
        # query each valid point's neighbours within neighbor_dist
        for j in valid_ids:
            nn = tree.query_ball_point(pts[t, j], neighbor_dist)
            if len(nn) < min_neighbors:
                mv[t, j] = False
                continue
            nbr_ids = valid_ids[nn]
            diff = np.linalg.norm(motions[t, j] - motions[t, nbr_ids], axis=1)
            if (diff < half).sum() < 0.5 * len(nbr_ids):
                mv[t, j] = False
    return mv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--camera_idx", type=int, default=0)
    parser.add_argument("--controller_name", default=CONTROLLER_NAME)
    parser.add_argument("--neighbor_dist", type=float, default=0.03,
                        help="Spatial neighbour radius (m) used by motion-consistency filter.")
    args = parser.parse_args()

    case_dir = os.path.join(args.base_path, args.case_name)
    cam = args.camera_idx
    cupid_dir = os.path.join(args.results_dir, args.case_name, "cupid")

    pose_json = os.path.join(cupid_dir, "pose.json")
    with open(pose_json) as f:
        pose = json.load(f)
    extrinsic = np.asarray(pose["extrinsic"], dtype=np.float64)
    intrinsic = np.asarray(pose["intrinsic"], dtype=np.float64)
    c2w = np.linalg.inv(extrinsic)

    mesh_path = find_mesh(cupid_dir)
    print(f"[lift_tracks_single] loading mesh: {mesh_path}")
    mesh = load_trimesh(mesh_path)

    track_npz = os.path.join(case_dir, "cotracker", f"{cam}.npz")
    data = np.load(track_npz)
    tracks_yx = data["tracks"].astype(np.float32)         # (T, N, 2)
    visibility = data["visibility"].astype(bool)          # (T, N)
    T, N, _ = tracks_yx.shape
    print(f"[lift_tracks_single] tracks {tracks_yx.shape}")

    frames = iio.imread(os.path.join(case_dir, "color", f"{cam}.mp4"), plugin="FFMPEG")
    H, W = frames.shape[1:3]
    print(f"[lift_tracks_single] video HxW = {H}x{W}")

    mask_info_path = os.path.join(case_dir, "mask", f"mask_info_{cam}.json")
    with open(mask_info_path) as f:
        mask_info = json.load(f)
    obj_id = None
    ctrl_ids = []
    for k, name in mask_info.items():
        if name == args.controller_name:
            ctrl_ids.append(int(k))
        else:
            if obj_id is not None:
                raise ValueError(
                    f"more than one non-controller mask present in {mask_info_path}: "
                    f"got at least {obj_id} and {k}"
                )
            obj_id = int(k)
    if obj_id is None:
        raise RuntimeError("no object mask found")
    object_mask0 = load_mask(os.path.join(case_dir, "mask", str(cam), str(obj_id), "0.png"))
    controller_mask0 = None
    if ctrl_ids:
        for cid in ctrl_ids:
            m = load_mask(os.path.join(case_dir, "mask", str(cam), str(cid), "0.png"))
            controller_mask0 = m if controller_mask0 is None else (controller_mask0 | m)
        print(f"[lift_tracks_single] unioned {len(ctrl_ids)} controller masks "
              f"({ctrl_ids})  -> {int(controller_mask0.sum())} pixels")

    # First-frame ray cast.
    py0 = tracks_yx[0, :, 0]
    px0 = tracks_yx[0, :, 1]
    rays_cam = build_rays_cam(px0, py0, W, H, intrinsic)            # (N, 3)
    rays_world = (c2w[:3, :3] @ rays_cam.T).T
    rays_world /= np.linalg.norm(rays_world, axis=1, keepdims=True)
    origin = c2w[:3, 3][None, :].repeat(N, axis=0)
    print(f"[lift_tracks_single] ray-casting {N} rays against mesh ...")
    hits_world, valid_first = cast_first_frame(mesh, origin, rays_world)
    print(f"[lift_tracks_single] {int(valid_first.sum())}/{N} rays hit the mesh")

    # Depth (z in camera frame) from frame 0.
    hits_cam = (extrinsic[:3, :3] @ hits_world.T).T + extrinsic[:3, 3]
    z0 = hits_cam[:, 2]

    # Frame-0 pixel-space splits we need for the fallback below.
    py0_int = np.clip(np.round(py0).astype(int), 0, H - 1)
    px0_int = np.clip(np.round(px0).astype(int), 0, W - 1)
    object_pixel = object_mask0[py0_int, px0_int]
    object_idx_all = np.where(object_pixel)[0]
    object_hit_mask = object_pixel & valid_first

    # For object-mask queries whose ray missed the mesh (silhouette edges,
    # disconnected components, mesh holes), inherit z0 from the closest
    # ray-hit OBJECT query in pixel space. This keeps the canonical "lift
    # along ray with a fixed z0" assumption while filling coverage holes
    # that Cupid mesh imperfections create.
    n_obj_total = int(object_pixel.sum())
    n_obj_hit = int(object_hit_mask.sum())
    n_obj_missed = n_obj_total - n_obj_hit
    if n_obj_missed > 0 and n_obj_hit > 0:
        from scipy.spatial import cKDTree
        hit_obj_idx = np.where(object_hit_mask)[0]
        missed_obj_idx = np.where(object_pixel & ~valid_first)[0]
        kd = cKDTree(np.stack([px0_int[hit_obj_idx], py0_int[hit_obj_idx]], axis=1))
        _, nn = kd.query(
            np.stack([px0_int[missed_obj_idx], py0_int[missed_obj_idx]], axis=1),
            k=1,
        )
        z0[missed_obj_idx] = z0[hit_obj_idx[nn]]
        valid_first[missed_obj_idx] = True
        print(
            f"[lift_tracks_single] filled {n_obj_missed}/{n_obj_total} object-mask "
            f"queries that missed the mesh with nearest-hit z0"
        )

    # Optional per-frame depth from Depth Anything (data/<case>/depth/<cam>/*.npy
    # in millimeters). Falls back to constant-z0 lift if absent.
    depth_dir = os.path.join(case_dir, "depth", str(cam))
    use_per_frame_depth = False
    if os.path.isdir(depth_dir) and glob.glob(os.path.join(depth_dir, "*.npy")):
        use_per_frame_depth = True
        # Scale relative to frame-0 anchor: keep z0 (from mesh ray-cast) as the
        # baseline depth per track, and modulate by the *ratio* of DA depth at
        # the current pixel to DA depth at the frame-0 pixel. This preserves
        # the high-fidelity per-track depth from the Cupid mesh while letting
        # DA contribute the across-frame depth motion that constant-z misses.
        print(f"[lift_tracks_single] using per-frame DA depth from {depth_dir}")

    # Lift every frame.
    points_world = np.full((T, N, 3), np.nan, dtype=np.float32)
    if use_per_frame_depth:
        # Cache frame-0 DA depth at every query pixel for the ratio.
        depth0 = np.load(os.path.join(depth_dir, "0.npy")).astype(np.float32) / 1000.0  # meters
        z_da0 = depth0[py0_int, px0_int]
        # Avoid division by ~0
        z_da0 = np.where(z_da0 > 1e-3, z_da0, np.nan)
    for t in range(T):
        py = tracks_yx[t, :, 0]
        px = tracks_yx[t, :, 1]
        py_i = np.clip(np.round(py).astype(int), 0, H - 1)
        px_i = np.clip(np.round(px).astype(int), 0, W - 1)
        rays_cam_t = build_rays_cam(px, py, W, H, intrinsic)        # (N, 3)
        if use_per_frame_depth:
            depth_t_path = os.path.join(depth_dir, f"{t}.npy")
            if os.path.isfile(depth_t_path):
                d_t = np.load(depth_t_path).astype(np.float32) / 1000.0  # meters
                z_da_t = d_t[py_i, px_i]
                ratio = np.where((z_da_t > 1e-3) & np.isfinite(z_da0),
                                 z_da_t / z_da0, 1.0)
                z_use = z0 * ratio
            else:
                z_use = z0
        else:
            z_use = z0
        # Cam-frame point along ray at the chosen z.
        cam_pt = rays_cam_t * z_use[:, None]                          # (N, 3)
        # Cam -> model (world) via c2w.
        world_pt = (c2w[:3, :3] @ cam_pt.T).T + c2w[:3, 3]
        points_world[t] = world_pt.astype(np.float32)
    points_world[:, ~valid_first] = np.nan

    # Sample frame-0 colors at query pixels.
    colors = frames[0][py0_int, px0_int].astype(np.float32) / 255.0  # (N, 3) RGB

    # Split by frame-0 mask membership.
    object_idx = np.where(
        valid_first
        & object_mask0[py0_int, px0_int]
    )[0]
    if controller_mask0 is not None:
        controller_idx = np.where(controller_mask0[py0_int, px0_int])[0]
    else:
        controller_idx = np.array([], dtype=int)

    # Per-frame motion_valid with neighbor-consistency filter, mirroring the
    # heuristic in data_process_track.py (multi-view pipeline). Starts from
    # visibility[t] & visibility[t+1] then drops points whose motion disagrees
    # with most of their spatial neighbors.
    obj_pts = points_world[:, object_idx]                              # (T, N_obj, 3)
    obj_vis = visibility[:, object_idx].astype(bool)                   # (T, N_obj)
    motions_valid = _compute_motions_valid_neighbor(
        obj_pts, obj_vis,
        neighbor_dist=args.neighbor_dist,
        min_neighbors=5,
    )

    object_out = os.path.join(case_dir, "object_tracks_3d.npz")
    np.savez(
        object_out,
        points=obj_pts.astype(np.float32),
        visibility=obj_vis,
        motions_valid=motions_valid,
        colors=colors[object_idx],
        valid_first=valid_first[object_idx],
    )
    print(f"[lift_tracks_single] wrote {object_out}  N_obj={object_idx.shape[0]}")
    n_basic = int((obj_vis[:-1] & obj_vis[1:]).sum())
    n_after = int(motions_valid[:-1].sum())
    print(f"  motions_valid: basic={n_basic} -> after neighbor filter={n_after}")

    if controller_idx.size > 0:
        # For controller pixels we use mid-frame plane: same constant-z approx,
        # but the ray-cast on the mesh wasn't designed for hand. We fall back to
        # the same back-projection but with z0 estimated from the median object
        # depth (so the hand sits at roughly the same depth as the object).
        if np.any(valid_first[object_idx]):
            z_ctrl = float(np.nanmedian(z0[object_idx][valid_first[object_idx]]))
        else:
            z_ctrl = 1.0
        rays_cam_first = build_rays_cam(px0[controller_idx], py0[controller_idx], W, H, intrinsic)
        z_ctrl_arr = np.full((controller_idx.size,), z_ctrl, dtype=np.float64)
        controller_world = np.full((T, controller_idx.size, 3), np.nan, dtype=np.float32)
        for t in range(T):
            py = tracks_yx[t, controller_idx, 0]
            px = tracks_yx[t, controller_idx, 1]
            rays_cam_t = build_rays_cam(px, py, W, H, intrinsic)
            cam_pt = rays_cam_t * z_ctrl_arr[:, None]
            world_pt = (c2w[:3, :3] @ cam_pt.T).T + c2w[:3, 3]
            controller_world[t] = world_pt.astype(np.float32)
        ctrl_out = os.path.join(case_dir, "controller_tracks_3d.npz")
        np.savez(
            ctrl_out,
            points=controller_world,
            visibility=visibility[:, controller_idx],
        )
        print(f"[lift_tracks_single] wrote {ctrl_out}  N_ctrl={controller_idx.shape[0]}")
    else:
        print("[lift_tracks_single] no controller mask -> skipping controller export")


if __name__ == "__main__":
    main()
