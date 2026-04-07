#!/usr/bin/env python3
"""
Load one case, build object + hand-object springs, visualize with Open3D.
- Mouse: rotate/pan/zoom camera.
- Press 's': save screenshot to screenshot_XXX.png (in --screenshot_dir).

Optionally save a PLY with --out for use in MeshLab etc.

Usage:
  python semantic/vis_springs_points.py --data_path /path/to/final_data.pkl [--out semantic/vis_out/springs.ply]
  python semantic/vis_springs_points.py --data_path /path/to/final_data.pkl --no_ply   # only Open3D vis, no PLY
"""

import argparse
import os
import sys
import numpy as np
import pickle
import open3d as o3d


OBJECT_KNN = 20
CONTROLLER_KNN = 30
OBJECT_RADIUS = 0.02
OBJECT_MAX_NEIGHBOURS = 30
CONTROLLER_RADIUS = 0.04
CONTROLLER_MAX_NEIGHBOURS = 50
USE_KNN_TOPOLOGY = True


def build_topology(
    object_points,
    controller_points=None,
    use_knn_topology=USE_KNN_TOPOLOGY,
    object_knn=OBJECT_KNN,
    controller_knn=CONTROLLER_KNN,
    object_radius=OBJECT_RADIUS,
    object_max_neighbours=OBJECT_MAX_NEIGHBOURS,
    controller_radius=CONTROLLER_RADIUS,
    controller_max_neighbours=CONTROLLER_MAX_NEIGHBOURS,
):
    object_points = np.asarray(object_points, dtype=np.float64)
    if controller_points is not None:
        controller_points = np.asarray(controller_points, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    n_object_points = len(points)

    spring_flags = np.zeros((len(points), len(points)))
    springs = []
    for i in range(len(points)):
        if use_knn_topology:
            k = min(object_knn + 1, len(points))
            [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k)
            idx = idx[1:]
        else:
            [_, idx, _] = pcd_tree.search_hybrid_vector_3d(
                points[i], object_radius, object_max_neighbours
            )
            idx = idx[1:]
        for j in idx:
            rest_length = np.linalg.norm(points[i] - points[j])
            if (
                spring_flags[i, j] == 0
                and spring_flags[j, i] == 0
                and rest_length > 1e-4
            ):
                spring_flags[i, j] = 1
                spring_flags[j, i] = 1
                springs.append([i, j])

    num_object_springs = len(springs)

    if controller_points is not None and len(controller_points) > 0:
        points = np.concatenate([points, controller_points], axis=0)
        for i in range(len(controller_points)):
            if use_knn_topology:
                k = min(controller_knn, n_object_points)
                [_, idx, _] = pcd_tree.search_knn_vector_3d(controller_points[i], k)
            else:
                [_, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    controller_points[i],
                    controller_radius,
                    controller_max_neighbours,
                )
            for j in idx:
                springs.append([n_object_points + i, j])

    springs = np.array(springs, dtype=np.int32)
    return points, springs, num_object_springs


def load_data_from_pickle(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    if "object_points" not in data:
        raise KeyError("Need object_points in pickle")
    object_points = data["object_points"]
    object_first = object_points[0] if object_points.ndim == 3 else object_points
    other_surface = data.get("surface_points", np.zeros((0, 3)))
    interior = data.get("interior_points", np.zeros((0, 3)))
    structure_points = np.concatenate(
        [object_first, other_surface, interior], axis=0
    ).astype(np.float64)
    controller_points = data.get("controller_points", None)
    if controller_points is not None:
        controller_points = (
            controller_points[0] if controller_points.ndim == 3 else controller_points
        )
        controller_points = np.asarray(controller_points, dtype=np.float64)
    else:
        controller_points = np.zeros((0, 3))
    return structure_points, controller_points


def _cube_vertices(center, half):
    """8 corners of a cube centered at center with half-extent half."""
    o = np.array(center)
    h = float(half)
    return o + np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ])


def _edge_quad_vertices(a, b, thickness):
    """4 vertices of a thin quad from a to b; thickness is half-width."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    L = np.linalg.norm(d) + 1e-12
    d = d / L
    # perpendicular: cross with smallest axis
    if abs(d[0]) <= abs(d[1]) and abs(d[0]) <= abs(d[2]):
        perp = np.cross(d, [1, 0, 0])
    else:
        perp = np.cross(d, [0, 1, 0])
    perp = perp / (np.linalg.norm(perp) + 1e-12) * thickness
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.array([
        a + perp, a - perp, b - perp, b + perp,
    ])


def write_ply(path, points, point_rgb, edges, edge_rgb, point_scale=0.003, edge_thickness=0.001):
    """Write one PLY: points as small cubes (vertex-colored), edges as thin quads (vertex-colored). MeshLab shows both."""
    points = np.asarray(points, dtype=np.float64)
    n_pts = len(points)
    n_edges = len(edges)
    point_rgb = np.clip(point_rgb, 0, 1)
    edge_rgb = np.clip(edge_rgb, 0, 1)

    # bbox for optional scale
    if n_pts > 0:
        diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0)) + 1e-9
        if edge_thickness <= 0:
            edge_thickness = diag * 0.0005
        if point_scale <= 0:
            point_scale = diag * 0.002

    # Point cubes: 8 verts + 12 faces each
    verts_list = []
    rgb_list = []
    for i in range(n_pts):
        v = _cube_vertices(points[i], point_scale)
        verts_list.append(v)
        rgb_list.append(np.tile(point_rgb[i], (8, 1)))
    verts_pts = np.vstack(verts_list)
    rgb_pts = np.vstack(rgb_list)
    # 12 faces per cube (two triangles per face)
    base = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)
    faces_pts = np.concatenate([base + 8 * i for i in range(n_pts)], axis=0)

    # Edge quads: 4 verts + 2 faces each
    n_pts_verts = len(verts_pts)
    for k in range(n_edges):
        i, j = edges[k, 0], edges[k, 1]
        q = _edge_quad_vertices(points[i], points[j], edge_thickness)
        verts_list.append(q)
        rgb_list.append(np.tile(edge_rgb[k], (4, 1)))
    verts_edges = np.vstack(verts_list[n_pts:]) if verts_list[n_pts:] else np.zeros((0, 3))
    rgb_edges = np.vstack(rgb_list[n_pts:]) if rgb_list[n_pts:] else np.zeros((0, 3))

    all_verts = np.vstack([verts_pts, verts_edges]) if len(verts_edges) > 0 else verts_pts
    all_rgb = np.vstack([rgb_pts, rgb_edges]) if len(rgb_edges) > 0 else rgb_pts
    # face indices for edge quads: 0,1,2 and 0,2,3
    if n_edges > 0:
        faces_edges = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        faces_edges = np.concatenate(
            [faces_edges + n_pts_verts + 4 * k for k in range(n_edges)], axis=0
        )
        all_faces = np.vstack([faces_pts, faces_edges])
    else:
        all_faces = faces_pts

    rgb_uint8 = (all_rgb * 255).astype(np.uint8)
    n_vertices = len(all_verts)
    n_faces = len(all_faces)

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar diffuse_red\nproperty uchar diffuse_green\nproperty uchar diffuse_blue\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(n_vertices):
            f.write(
                f"{all_verts[i,0]} {all_verts[i,1]} {all_verts[i,2]} "
                f"{int(rgb_uint8[i,0])} {int(rgb_uint8[i,1])} {int(rgb_uint8[i,2])}\n"
            )
        for t in range(n_faces):
            tri = all_faces[t]
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
    print(f"Saved: {path} ({n_vertices} vertices, {n_faces} faces; points as cubes, edges as quads)")


def build_o3d_geometries(points, point_rgb, edges, edge_rgb):
    """Build Open3D PointCloud and LineSet for visualization."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(point_rgb, dtype=np.float64), 0, 1))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(edge_rgb, dtype=np.float64), 0, 1))
    return pcd, line_set


def run_o3d_visualizer(pcd, line_set, screenshot_dir, point_size=4.0):
    """Run Open3D visualizer: mouse for camera, 's' to save screenshot."""
    os.makedirs(screenshot_dir, exist_ok=True)
    screenshot_counter = [0]  # mutable for callback

    def save_screenshot_callback(vis):
        screenshot_counter[0] += 1
        path = os.path.join(screenshot_dir, f"screenshot_{screenshot_counter[0]:04d}.png")
        try:
            vis.capture_screen_image(path, do_render=True)
        except TypeError:
            vis.capture_screen_image(path)
        print(f"Saved screenshot: {path}")
        return False  # keep running

    geometries = [pcd, line_set]
    key_to_callback = {
        ord("s"): save_screenshot_callback,
        ord("S"): save_screenshot_callback,
    }
    o3d.visualization.draw_geometries_with_key_callbacks(
        geometries,
        key_to_callback,
        window_name="Springs & points (mouse: rotate/pan/zoom, s: screenshot)",
        width=1280,
        height=720,
        left=50,
        top=50,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
        "--out",
        type=str,
        default="semantic/vis_out/springs.ply",
        help="Output PLY path",
    )
    parser.add_argument(
        "--point_scale",
        type=float,
        default=0.004,
        help="Half-size of point cubes (larger = bigger points). Default 0.004.",
    )
    parser.add_argument(
        "--edge_thickness",
        type=float,
        default=0.001,
        help="Half-width of edge quads. Use 0 for auto from bbox.",
    )
    parser.add_argument(
        "--no_ply",
        action="store_true",
        help="Do not write PLY file; only run Open3D visualization.",
    )
    parser.add_argument(
        "--screenshot_dir",
        type=str,
        default="semantic/vis_out/screenshots",
        help="Directory for screenshots when pressing 's'. Default: semantic/vis_out/screenshots",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=4.0,
        help="Point size in Open3D window. Default 4.0.",
    )
    args = parser.parse_args()

    if args.data_path is None or not os.path.isfile(args.data_path):
        for base in ["results", "qqtt", "."]:
            for root, _, files in os.walk(base):
                for f in files:
                    if f == "final_data.pkl":
                        args.data_path = os.path.join(root, f)
                        break
                if args.data_path is not None:
                    break
            if args.data_path is not None:
                break
    if args.data_path is None or not os.path.isfile(args.data_path):
        print("Use --data_path /path/to/final_data.pkl")
        sys.exit(1)

    structure_points, controller_points = load_data_from_pickle(args.data_path)
    points, springs, num_object_springs = build_topology(
        structure_points,
        controller_points
        if (controller_points is not None and controller_points.size > 0)
        else None,
    )
    n_object_points = len(structure_points)
    n_controller = points.shape[0] - n_object_points

    # Vertex colors: structure blue, controller red (bright so MeshLab shows them)
    color_structure = np.array([0.3, 0.5, 1.0])
    color_controller = np.array([1.0, 0.2, 0.2])
    point_rgb = np.tile(color_structure, (n_object_points, 1))
    if n_controller > 0:
        point_rgb = np.concatenate(
            [point_rgb, np.tile(color_controller, (n_controller, 1))], axis=0
        )

    # Edge colors: object springs green, hand-object orange
    color_object = np.array([0.2, 0.9, 0.3])
    color_hand = np.array([1.0, 0.55, 0.0])
    object_edges = springs[:num_object_springs]
    hand_edges = springs[num_object_springs:]
    edge_rgb = np.tile(color_object, (len(object_edges), 1))
    if len(hand_edges) > 0:
        edge_rgb = np.concatenate(
            [edge_rgb, np.tile(color_hand, (len(hand_edges), 1))], axis=0
        )
    all_edges = springs

    # Open3D visualization (mouse: camera pose, 's': screenshot)
    pcd, line_set = build_o3d_geometries(points, point_rgb, all_edges, edge_rgb)
    run_o3d_visualizer(pcd, line_set, args.screenshot_dir, point_size=args.point_size)

    # Optionally write PLY
    if not args.no_ply:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        write_ply(
            args.out,
            points,
            point_rgb,
            all_edges,
            edge_rgb,
            point_scale=args.point_scale,
            edge_thickness=args.edge_thickness,
        )


if __name__ == "__main__":
    main()
