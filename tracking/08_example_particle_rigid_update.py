import numpy as np
import os
import imageio.v2 as imageio
import argparse
import cv2
import torch
import torch.nn.functional as F
import sys
import open3d as o3d
import json

from pathlib import Path
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import tempfile
import shutil

from torch.utils.data import DataLoader
from torchvision import transforms
import open_clip

# local co-tracker repo (for imports etc. if needed)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "co-tracker"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mapping.mapping_lib.implicit_decoder import ImplicitDecoder
from mapping.mapping_lib.voxel_hash_table import VoxelHashTable
from mapping.mapping_lib.vision_wrapper import EvaClipWrapper, DINOv3Wrapper
from mapping.mapping_lib.utils import get_3d_coordinates


def draw_geometries_with_key_callbacks(
    geometries,
    window_name="Open3D",
    camera_extrinsic=None,
):
    """
    Custom Open3D visualizer that supports:
    - Registering a key callback to print the camera pose (press 'P').
    - Setting an initial camera pose.
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name)

    for geometry in geometries:
        vis.add_geometry(geometry)

    def print_camera_pose(vis):
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = np.asarray(params.extrinsic)
        print(f"\n[{window_name}] Camera Extrinsic Pose (copy-paste this):")
        print(f"camera_extrinsic = {repr(extrinsic)}")
        return False

    vis.register_key_callback(ord("P"), print_camera_pose)

    if camera_extrinsic is not None:
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        params.extrinsic = camera_extrinsic
        ctr.convert_from_pinhole_camera_parameters(params, False)

    vis.run()
    vis.destroy_window()


# ==============================================================================================
#  3D point from depth + pose (single pixel)
# ==============================================================================================

def compute_3d_point_from_depth(
    u: float,
    v: float,
    depth_map: np.ndarray,
    pose_world_to_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    """
    Computes the 3D point in the world coordinate system from a given pixel (u, v)
    using depth, pose, and intrinsics.
    pose_world_to_cam: 4x4, world -> cam
    """
    h, w = depth_map.shape
    u_i = int(round(u))
    v_i = int(round(v))

    if u_i < 0 or u_i >= w or v_i < 0 or v_i >= h:
        return None

    z = depth_map[v_i, u_i]
    if z <= 0:
        return None

    # camera coordinate system
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    p_cam = np.array([x_cam, y_cam, z], dtype=np.float32)

    # cam -> world transformation
    pose_cam_to_world = np.linalg.inv(pose_world_to_cam)
    R_c2w = pose_cam_to_world[:3, :3]
    t_c2w = pose_cam_to_world[:3, 3]
    p_world = R_c2w @ p_cam + t_c2w
    return p_world


# ==============================================================================================
#  Dense RGB point cloud from depth + pose (whole image)
# ==============================================================================================

def compute_dense_point_cloud_from_depth_and_rgb(
    depth_map: np.ndarray,
    rgb_frame: np.ndarray,
    pose_world_to_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 4,
):
    """
    Compute dense 3D point cloud for the whole image (with optional stride),
    using depth, pose, and intrinsics.

    Returns:
        pts_world: (M, 3) float32
        colors:   (M, 3) uint8 (RGB)
    """
    h, w = depth_map.shape

    # sample pixels with stride (to avoid huge point clouds)
    vs, us = np.mgrid[0:h:stride, 0:w:stride]  # (h_s, w_s)
    z = depth_map[vs, us]  # (h_s, w_s)

    valid_mask = z > 0
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    us_valid = us[valid_mask].astype(np.float32)
    vs_valid = vs[valid_mask].astype(np.float32)
    z_valid = z[valid_mask].astype(np.float32)

    x_cam = (us_valid - cx) * z_valid / fx
    y_cam = (vs_valid - cy) * z_valid / fy
    pts_cam = np.stack([x_cam, y_cam, z_valid], axis=1)  # (M, 3)

    pose_cam_to_world = np.linalg.inv(pose_world_to_cam)
    R_c2w = pose_cam_to_world[:3, :3]
    t_c2w = pose_cam_to_world[:3, 3]
    pts_world = (R_c2w @ pts_cam.T).T + t_c2w  # (M, 3)

    # RGB
    rgb_sampled = rgb_frame[vs, us, :]  # (h_s, w_s, 3)
    colors = rgb_sampled[valid_mask].reshape(-1, 3).astype(np.uint8)

    return pts_world.astype(np.float32), colors


# ==============================================================================================
#  Multi-keypoint selection (mouse)
# ==============================================================================================

def select_multiple_keypoints(rgb_image: np.ndarray):
    """
    Allows the user to select MULTIPLE keypoints with the mouse on the first frame.
    - Left click: add a keypoint
    - Right click: remove the last keypoint
    - ENTER: confirm selection
    - ESC: cancel (returns empty list)

    Returns:
        list of (u, v) tuples
    """
    keypoints = []
    img_disp = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    img_copy = img_disp.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal img_copy
        if event == cv2.EVENT_LBUTTONDOWN:
            keypoints.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if keypoints:
                keypoints.pop()
        
        # Redraw
        img_copy = img_disp.copy()
        for i, pt in enumerate(keypoints):
            # Draw a circle
            cv2.circle(img_copy, pt, 5, (0, 0, 255), -1)
            # Optionally number them
            cv2.putText(img_copy, str(i), (pt[0]+5, pt[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("Select Keypoints", img_copy)

    cv2.namedWindow("Select Keypoints", cv2.WINDOW_NORMAL)
    cv2.imshow("Select Keypoints", img_copy)
    cv2.setMouseCallback("Select Keypoints", mouse_callback)

    print("[INFO] Left-click to ADD points. Right-click to REMOVE last. ENTER to confirm. ESC to cancel.")
    print("[INFO] NOTE: The FIRST point selected will be used to define the object (ball query).")
    print("[INFO]       ALL points will be used to estimate Rigid Body Motion (SE3).")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            keypoints = []
            break
        if key in (13, 10):  # ENTER
            break

    cv2.destroyWindow("Select Keypoints")
    return keypoints


# ==============================================================================================
#  CoTracker3 ONLINE tracking for multiple keypoints
# ==============================================================================================

def run_cotracker_online_multi(
    frames_np: np.ndarray,
    keypoints: list[tuple[float, float]],
    device: str = "cuda",
):
    """
    Run CoTracker3 in ONLINE mode for multiple user-selected keypoints.

    frames_np: (T, H, W, 3) uint8 RGB
    keypoints: list of (u, v) on the FIRST frame in this clip (x=u, y=v)

    Returns:
        tracks_xy: (T, N, 2) numpy array of (x, y) per frame and keypoint
        vis_np   : (T, N) numpy array of visibility/confidence per frame and keypoint
    """
    if len(keypoints) == 0:
        raise ValueError("No keypoints provided to run_cotracker_online_multi.")

    # (T, H, W, 3) -> (1, T, 3, H, W)
    video = torch.from_numpy(frames_np).permute(0, 3, 1, 2)[None].float().to(device)
    T = video.shape[1]

    # Load online model from torch.hub
    print("[CoTracker] Loading CoTracker3 online model from torch.hub ...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
    cotracker.eval()
    print(f"[CoTracker] step={cotracker.step}")

    # Multiple query points on the first frame of this clip: (t=0, x=u, y=v)
    queries_list = [[0.0, float(u), float(v)] for (u, v) in keypoints]
    queries = torch.tensor(
        [queries_list],  # (B=1, N, 3)
        dtype=torch.float32,
        device=device,
    )

    grid_size = 0  # only user-selected queries, no dense grid

    with torch.inference_mode():
        # 1) Initialize online tracking (is_first_step=True) - queries MUST NOT be None
        cotracker(
            video_chunk=video,
            is_first_step=True,
            grid_size=grid_size,
            queries=queries,
            add_support_grid=True,
        )

        step = cotracker.step

        # 2) Actual inference with sliding windows (T must be <= window size for non-first steps)
        if T <= 2 * step:
            # Short clip: can process in one go
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video,
                grid_size=grid_size,
                queries=queries,
                add_support_grid=True,
            )
        else:
            pred_tracks = None
            pred_visibility = None
            for ind in range(0, T - step, step):
                chunk = video[:, ind: ind + 2 * step]  # length <= 2*step
                pred_tracks, pred_visibility = cotracker(
                    video_chunk=chunk,
                    grid_size=grid_size,
                    queries=queries,
                    add_support_grid=True,
                )
                # online mode maintains state; final pred_tracks assumed to contain full track

    # pred_tracks: (B, T, N, 2), pred_visibility: (B, T, N[, 1])
    tracks_xy = pred_tracks[0].detach().cpu().numpy()  # (T, N, 2)

    vis = pred_visibility[0]  # (T, N) or (T, N, 1)
    if vis.dim() == 3:
        vis = vis[..., 0]
    vis_np = vis.detach().cpu().numpy()  # (T, N)

    return tracks_xy, vis_np


# ==============================================================================================
#  3D positions for keypoints + RGB colors over time (T, N, 3)
#  + dense RGB point cloud for each time step
# ==============================================================================================

def compute_3d_positions_and_colors_over_time(
    frames_np: np.ndarray,       # (T, H, W, 3) RGB
    tracks_xy: np.ndarray,       # (T, N, 2)
    visibilities: np.ndarray,    # (T, N)
    depth_files: list[str],
    pose_files: list[str],
    start_frame_index: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    vis_threshold: float = 0.8,
    point_stride: int = 4,       # stride for dense point cloud
):
    """
    Compute 3D positions and RGB colors for each keypoint at each time step,
    AND dense RGB point cloud for the entire image at each time step.

    Returns:
        positions_3d: (T_eff, N, 3) with NaN where invalid (keypoints)
        colors_rgb  : (T_eff, N, 3) uint8, with 0 where invalid
        pc_positions: list of length T_eff, each (M_t, 3) dense point cloud
        pc_colors   : list of length T_eff, each (M_t, 3) uint8 RGB
    """
    T, N, _ = tracks_xy.shape
    T_eff = min(
        T,
        visibilities.shape[0],
        frames_np.shape[0],
        len(depth_files) - start_frame_index,
        len(pose_files) - start_frame_index,
    )
    tracks_xy = tracks_xy[:T_eff]
    visibilities = visibilities[:T_eff]
    frames_np = frames_np[:T_eff]

    positions_3d = np.full((T_eff, N, 3), np.nan, dtype=np.float32)
    colors_rgb = np.zeros((T_eff, N, 3), dtype=np.uint8)

    pc_positions = []
    pc_colors = []

    print("[3D] Computing 3D positions (keypoints) + RGB colors + dense point clouds over time...")
    for t in range(T_eff):
        frame_idx = start_frame_index + t
        depth = np.load(depth_files[frame_idx])
        pose_world_to_cam = np.load(pose_files[frame_idx])
        rgb_frame = frames_np[t]  # (H, W, 3), uint8

        H, W, _ = rgb_frame.shape

        # Dense RGB point cloud (whole image)
        pts_pc, cols_pc = compute_dense_point_cloud_from_depth_and_rgb(
            depth_map=depth,
            rgb_frame=rgb_frame,
            pose_world_to_cam=pose_world_to_cam,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            stride=point_stride,
        )
        pc_positions.append(pts_pc)
        pc_colors.append(cols_pc)

        # Keypoint 3D trajectories
        for n in range(N):
            if visibilities[t, n] < vis_threshold:
                continue

            u, v = tracks_xy[t, n]
            p_world = compute_3d_point_from_depth(
                u=u,
                v=v,
                depth_map=depth,
                pose_world_to_cam=pose_world_to_cam,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )

            if p_world is None:
                continue

            positions_3d[t, n, :] = p_world

            # RGB at keypoint pixel
            u_i = int(round(u))
            v_i = int(round(v))
            if 0 <= u_i < W and 0 <= v_i < H:
                colors_rgb[t, n, :] = rgb_frame[v_i, u_i, :]

    return positions_3d, colors_rgb, pc_positions, pc_colors


# ==============================================================================================
#  Rigid Transform Estimation (SE3)
# ==============================================================================================

def solve_rigid_transform(A: np.ndarray, B: np.ndarray):
    """
    Finds R, t such that B approx R @ A + t
    A, B: (N, 3) corresponding points
    """
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    
    # H = AA^T @ BB
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    
    # R = V @ U.T
    R = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    t = centroid_B - R @ centroid_A
    return R, t


def estimate_transforms_over_time(positions_3d: np.ndarray):
    """
    positions_3d: (T, N, 3)
    Returns:
        transforms: list of (R, t) tuples for each frame 0..T-1
    """
    T, N, _ = positions_3d.shape
    transforms = []
    
    # Use t=0 as the reference frame
    pts_0 = positions_3d[0]
    valid_mask_0 = ~np.isnan(pts_0[:, 0])
    indices_0 = np.where(valid_mask_0)[0]
    
    if len(indices_0) == 0:
        print("[WARN] No valid keypoints at t=0. Cannot estimate transforms.")
        # Return identity transforms
        return [(np.eye(3), np.zeros(3)) for _ in range(T)]

    for t in range(T):
        pts_t = positions_3d[t]
        valid_mask_t = ~np.isnan(pts_t[:, 0])
        
        # Intersect valid indices: only use points valid in BOTH frame 0 and frame t
        common_indices = np.intersect1d(indices_0, np.where(valid_mask_t)[0])
        
        if len(common_indices) == 0:
            # Fallback: assume identity if lost (or use previous transform?)
            # Here we use identity or previous if available
            if len(transforms) > 0:
                transforms.append(transforms[-1])
            else:
                transforms.append((np.eye(3), np.zeros(3)))
            continue
        
        A = positions_3d[0, common_indices]
        B = positions_3d[t, common_indices]
        
        if len(common_indices) < 3:
             # Fewer than 3 points: SVD might be unstable for rotation.
             # If 1 point: translation only.
             if len(common_indices) == 1:
                 R = np.eye(3)
                 t_vec = B[0] - A[0]
             else:
                 # 2 points: ambiguous rotation around the axis connecting them.
                 # Still can try SVD, but result might drift.
                 R, t_vec = solve_rigid_transform(A, B)
        else:
            # Rigid transform (SVD)
            R, t_vec = solve_rigid_transform(A, B)
            
        transforms.append((R, t_vec))
        
    return transforms


# ==============================================================================================
#  Helper: compute features for arbitrary 3D points
# ==============================================================================================

def compute_features_for_points(
    points_3d: np.ndarray,
    grid: VoxelHashTable,
    decoder: ImplicitDecoder,
    device: str,
    batch_size: int = 100000,
) -> np.ndarray:
    """
    Query latent map (grid + decoder) for arbitrary 3D points.
    Returns:
        features_np: (N, C) numpy array
    """
    if points_3d.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    all_features = []
    with torch.no_grad():
        for i in range(0, points_3d.shape[0], batch_size):
            batch_points = points_3d[i:i+batch_size]
            pts_tensor = torch.from_numpy(batch_points).float().to(device)
            voxel_feat = grid.query_voxel_feature(pts_tensor, mark_accessed=False)
            feat = decoder(voxel_feat)  # (B, C)
            all_features.append(feat.cpu().numpy())

    features_np = np.concatenate(all_features, axis=0)
    return features_np


# ==============================================================================================
#  NEW: Ball query + latent feature cosine similarity + Y, f_y 반환
# ==============================================================================================

def ball_query_and_feature_similarity(
    keypoint_3d_world: np.ndarray,
    latent_points: np.ndarray,
    grid: VoxelHashTable,
    decoder: ImplicitDecoder,
    device: str,
    radius: float = 0.10,        # 10 cm
    cos_threshold: float = 0.8,  # threshold used for visualization of "high" sims in B(x)
):
    """
    1. Find the point x in latent_points (P) closest to keypoint_3d_world using KNN.
    2. Perform a ball query around x with a given radius (m) -> B(x).
    3. Compute F(x) and F(y) (for y in B(x)) using grid+decoder and calculate cosine similarity.
       - Points with cos(F(y), F(x)) >= cos_threshold are visualized as 'high sim'.
       - Concurrently, define Y as the set of points where cos(F(y), F(x)) >= 0.7.
    4. For points in Y, store their features F(y) in a separate tensor f_y.
    5. Visualize B(x) and the high-similarity points using Open3D.
    6. Returns:
         - dict["Y_indices"]: Indices into P (np.ndarray)
         - dict["f_y"]: Feature tensor F(y) (torch.Tensor)
         - Other debug info
    """
    Y_COS_THRESHOLD = 0.7  # Threshold for defining the set Y

    if latent_points.shape[1] > 3:
        latent_xyz = latent_points[:, :3]
    else:
        latent_xyz = latent_points

    print(f"[BALL] latent point cloud size = {latent_xyz.shape[0]}")

    # Build Open3D point cloud and KD-tree for KNN & radius search
    pcd_latent = o3d.geometry.PointCloud()
    pcd_latent.points = o3d.utility.Vector3dVector(latent_xyz.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(pcd_latent)

    query = keypoint_3d_world.astype(np.float64)

    # 1) KNN (k=1) to find x
    k, idx_knn, dist2_knn = kdtree.search_knn_vector_3d(query, 1)
    if k == 0:
        print("[BALL][ERROR] KNN search failed. Maybe keypoint is outside scene bounds?")
        return {
            "Y_indices": np.array([], dtype=np.int64),
            "f_y": torch.empty(0),
        }

    x_idx = idx_knn[0]
    x = latent_xyz[x_idx]
    print(f"[BALL] Nearest latent point index = {x_idx}, dist = {np.sqrt(dist2_knn[0]):.4f} m")
    print(f"[BALL] x (nearest latent point) = {x}")

    # 2) Ball query around x with radius
    k_ball, idx_ball, _ = kdtree.search_radius_vector_3d(x.astype(np.float64), radius)
    if k_ball == 0:
        print(f"[BALL][WARN] No points found within radius {radius} m around x.")
        return {
            "Y_indices": np.array([], dtype=np.int64),
            "f_y": torch.empty(0),
        }

    idx_ball = np.array(idx_ball, dtype=np.int64)  # indices into latent_points / P
    points_ball = latent_xyz[idx_ball]
    print(f"[BALL] |B(x)| (points within {radius*100:.1f} cm) = {points_ball.shape[0]}")

    # 3) Compute F(x) and F(y) for y ∈ B(x)
    with torch.no_grad():
        # F(x) (raw + normalized)
        x_tensor = torch.from_numpy(x[None, :]).float().to(device)  # (1, 3)
        voxel_feat_x = grid.query_voxel_feature(x_tensor, mark_accessed=False)
        feat_x = decoder(voxel_feat_x)              # (1, C)
        feat_x_norm = F.normalize(feat_x, dim=-1)   # for cosine

        batch_size = 8192
        sims_list = []
        features_ball_list = []

        for i in range(0, points_ball.shape[0], batch_size):
            batch_points = points_ball[i:i+batch_size]
            pts_tensor = torch.from_numpy(batch_points).float().to(device)  # (B, 3)
            voxel_feat = grid.query_voxel_feature(pts_tensor, mark_accessed=False)
            feat_batch = decoder(voxel_feat)             # (B, C) raw features
            feat_batch_norm = F.normalize(feat_batch, dim=-1)

            sims_batch = torch.matmul(feat_batch_norm, feat_x_norm[0])  # (B,)
            sims_list.append(sims_batch.cpu().numpy())
            features_ball_list.append(feat_batch.cpu().numpy())

    sims = np.concatenate(sims_list, axis=0)                   # (|B(x)|,)
    features_ball = np.concatenate(features_ball_list, axis=0) # (|B(x)|, C)
    assert sims.shape[0] == points_ball.shape[0]
    assert features_ball.shape[0] == points_ball.shape[0]

    # high sim for visualization (>= cos_threshold)
    mask_high = sims >= cos_threshold
    num_high = mask_high.sum()
    print(f"[BALL] #points with cosine similarity ≥ {cos_threshold} : {num_high}")

    # Y set: cos(F(y), F(x)) >= 0.7
    mask_Y = sims >= Y_COS_THRESHOLD
    num_Y = mask_Y.sum()
    print(f"[BALL] #points in Y (cos ≥ {Y_COS_THRESHOLD}) : {num_Y}")

    if num_Y > 0:
        f_y = torch.from_numpy(features_ball[mask_Y]).float().to(device)  # (|Y|, C)
        Y_indices_global = idx_ball[mask_Y]  # indices into full point cloud P
    else:
        # empty Y
        f_y = torch.empty((0, features_ball.shape[1]), dtype=torch.float32, device=device)
        Y_indices_global = np.array([], dtype=np.int64)

    # 4) Visualization with Open3D for B(x)
    colors = np.zeros((points_ball.shape[0], 3), dtype=np.float64)
    colors[:] = np.array([0.6, 0.6, 0.6])  # Gray: all points in B(x)

    # high similarity points (>= cos_threshold) -> Red
    colors[mask_high] = np.array([1.0, 0.0, 0.0])

    # Highlight x itself in green
    if x_idx in idx_ball:
        local_idx = np.where(idx_ball == x_idx)[0][0]
        colors[local_idx] = np.array([0.0, 1.0, 0.0])
    else:
        print("[BALL][WARN] x is not in B(x) indices (numerical issue?).")

    pcd_ball = o3d.geometry.PointCloud()
    pcd_ball.points = o3d.utility.Vector3dVector(points_ball.astype(np.float64))
    pcd_ball.colors = o3d.utility.Vector3dVector(colors)

    print("[BALL] Displaying ball query and cosine-filtered points in Open3D...")
    draw_geometries_with_key_callbacks(
        [pcd_ball],
        window_name=f"Ball query (r={radius} m) & cosine ≥ {cos_threshold}",
    )

    return {
        "x": x,
        "x_idx": x_idx,
        "idx_ball": idx_ball,
        "points_ball": points_ball,
        "sims": sims,
        "Y_indices": Y_indices_global,
        "f_y": f_y,
    }


# ==============================================================================================
#  ONLINE UPDATE + VISUALIZATION
# ==============================================================================================

def visualize_excluded_masks(
    rgb_files: list[str],
    seg_files: list[str],
    excluded_ids: list[int],
    output_path: str = "excluded_mask_visualization.mp4",
    fps: int = 10,
):
    """
    Visualizes the excluded masks (based on excluded_ids) overlaid on RGB images.
    """
    print(f"[VIS] Visualizing excluded masks to {output_path}...")
    
    if not rgb_files or not seg_files:
        print("[VIS] No files provided.")
        return

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with imageio.get_writer(output_path, fps=fps) as writer:
        for i, (rgb_path, seg_path) in enumerate(zip(rgb_files, seg_files)):
            if i % 10 == 0:
                print(f"[VIS] Processing frame {i}/{len(rgb_files)}", end='\r')
            
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            seg = np.load(seg_path)
            
            if not excluded_ids:
                mask = np.zeros_like(seg, dtype=bool)
            else:
                mask = np.isin(seg, excluded_ids)
            
            if np.any(mask):
                overlay = rgb.copy()
                overlay[mask] = [255, 0, 0] # Red
                alpha = 0.5
                rgb = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)
            
            writer.append_data(rgb)
            
    print(f"\\n[VIS] Done. Video saved to {output_path}")


def get_excluded_ids(json_path: str) -> list[int]:
    """
    Parses the instance_id_to_name.json and returns a list of IDs (integers)
    that correspond to 'teddy bear' or 'robot'.
    """
    if not os.path.exists(json_path):
        print(f"[WARN] Instance ID mapping file not found: {json_path}")
        return []

    with open(json_path, 'r') as f:
        mapping = json.load(f)

    excluded_ids = []
    for str_id, name in mapping.items():
        # Check for keywords
        # The name is a path like "/World/scene_0/teddy_bear_267/base_link/visuals"
        name_lower = name.lower()
        if "teddy_bear" in name_lower or "robot" in name_lower:
            try:
                excluded_ids.append(int(str_id))
            except ValueError:
                pass
    
    return excluded_ids


def update_latent_map_step(
    grid: VoxelHashTable,
    decoder: ImplicitDecoder,
    vision_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rgb_tensor: torch.Tensor,    # (1, 3, H, W) normalized
    depth_tensor: torch.Tensor,  # (1, 1, H_feat, W_feat)
    pose_world_to_cam: torch.Tensor, # (4, 4) or (1, 4, 4)
    seg_map: np.ndarray,         # (H, W)
    excluded_ids: list[int],
    intrinsics: tuple,           # (fx, fy, cx, cy)
    original_image_size: int,
    device: str,
    scene_bounds: tuple,         # (min, max)
    num_iterations: int = 1,
):
    """
    Performs one step of online update for the latent map.
    """
    grid.train()
    decoder.eval() # Frozen

    fx, fy, cx, cy = intrinsics
    scene_min, scene_max = scene_bounds

    # 1. Extract Features
    with torch.no_grad():
        vis_feat = vision_model(rgb_tensor) # (1, C, Hf, Wf)
    
    B, C_, Hf, Wf = vis_feat.shape

    # 2. Compute 3D Coordinates
    # pose_world_to_cam is W2C. We need extrinsic C2W? 
    # get_3d_coordinates usually expects extrinsic (C2W) or similar?
    # latent_map.py uses: E_cv = self.cam_to_world_poses... extrinsic_t = ...
    # So it expects CamToWorld.
    # The input pose_world_to_cam is W2C. Inverse it.
    if pose_world_to_cam.dim() == 2:
        pose_world_to_cam = pose_world_to_cam.unsqueeze(0) # (1, 4, 4)
    
    pose_c2w = torch.linalg.inv(pose_world_to_cam)
    extrinsic_t = pose_c2w[:, :3, :] # (1, 3, 4)

    coords_world, _ = get_3d_coordinates(
        depth_tensor, extrinsic_t,
        fx=fx, fy=fy, cx=cx, cy=cy,
        original_size=original_image_size,
    ) # (1, 3, Hf, Wf)

    # 3. Filter Points
    # Reshape
    feats_valid = vis_feat.permute(0, 2, 3, 1).reshape(-1, C_)
    coords_valid = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)

    # 3a. Scene Bounds
    in_x = (coords_valid[:,0] > scene_min[0]) & (coords_valid[:,0] < scene_max[0])
    in_y = (coords_valid[:,1] > scene_min[1]) & (coords_valid[:,1] < scene_max[1])
    in_z = (coords_valid[:,2] > scene_min[2]) & (coords_valid[:,2] < scene_max[2])
    in_bounds = in_x & in_y & in_z

    # 3b. Depth validation
    depth_flat = depth_tensor.reshape(-1)
    valid_depth = depth_flat >= 0.01
    in_bounds = in_bounds & valid_depth

    # 3c. Segmentation Mask (Moving Objects)
    # Resize seg_map to feature resolution (Hf, Wf)
    # seg_map is numpy (H, W).
    # We need to match the flattened indices.
    # Best to resize seg_map to (Hf, Wf) using nearest neighbor.
    seg_tensor = torch.from_numpy(seg_map).unsqueeze(0).unsqueeze(0).float().to(device) # (1, 1, H, W)
    seg_resized = F.interpolate(seg_tensor, size=(Hf, Wf), mode='nearest-exact') # (1, 1, Hf, Wf)
    seg_flat = seg_resized.reshape(-1).cpu().numpy().astype(int)
    
    # Identify mask of excluded pixels
    # This can be slow if excluded_ids is large. Use np.isin
    is_excluded = np.isin(seg_flat, excluded_ids)
    is_excluded_torch = torch.from_numpy(is_excluded).to(device)
    
    # Keep only NOT excluded
    mask_final = in_bounds & (~is_excluded_torch)

    if mask_final.sum() == 0:
        return None

    coords_train = coords_valid[mask_final] # (N_valid, 3)
    feats_target = feats_valid[mask_final]  # (N_valid, C)

    # 4. Optimization Step (Multiple iterations)
    for _ in range(num_iterations):
        optimizer.zero_grad()
        
        voxel_feat = grid.query_voxel_feature(coords_train)
        pred_feat = decoder(voxel_feat)

        # Cosine Similarity Loss
        # maximize cosine sim => minimize 1 - cos
        cos_sim = F.cosine_similarity(pred_feat, feats_target, dim=-1)
        loss = 1.0 - cos_sim.mean()

        loss.backward()
        optimizer.step()

    return coords_train.detach().cpu().numpy() # Return new points for visualization accumulation


def run_online_update_and_visualize(
    point_cloud_init: np.ndarray,
    grid: VoxelHashTable,
    decoder: ImplicitDecoder,
    vision_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    # Data sources
    rgb_files: list[str],
    depth_files: list[str],
    pose_files: list[str],
    seg_files: list[str],
    frame_indices: list[int], # The indices into the file lists that we are processing
    # Tracking results
    transforms_list: list[tuple], # (R, t) for each frame in frame_indices
    # Keypoint/Object info
    Y_indices: np.ndarray,
    f_y: torch.Tensor,
    # Configuration
    excluded_ids: list[int],
    intrinsics: tuple, # fx, fy, cx, cy
    original_image_size: int,
    scene_bounds: tuple,
    device: str,
    transform_func, # for image preprocessing
    feature_map_size: int, # Hf
    visualization_step: int = 30,
    output_video_path: str = "dynamic_pca.mp4",
    fps: int = 10,
    camera_extrinsic: np.ndarray = None,
    pca_batch_size: int = 100000,
    max_points: int = 2000000,
    update_steps: int = 1,
):
    # Initialize background point cloud (P \ Y)
    # Y_indices are indices in point_cloud_init
    N_init = point_cloud_init.shape[0]
    mask_Y = np.zeros(N_init, dtype=bool)
    mask_Y[Y_indices] = True
    
    # Start with initial points minus Y
    P_background_list = [point_cloud_init[~mask_Y]]
    
    # Y points at t=0
    pts_Y_init = point_cloud_init[Y_indices]
    
    print(f"[ONLINE] Starting online update loop. Frames: {len(frame_indices)}")
    print(f"[ONLINE] Visualization step: {visualization_step}")
    
    # Setup visualization recording
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Online Dynamic PCA (Recording)")
    temp_dir = tempfile.mkdtemp()
    frame_files_out = []
    pcd_dyn = o3d.geometry.PointCloud()
    
    try:
        for i, idx in enumerate(frame_indices):
            if i >= len(transforms_list):
                break
                
            # Load data
            rgb_path = rgb_files[idx]
            depth_path = depth_files[idx]
            pose_path = pose_files[idx]
            seg_path = seg_files[idx]

            rgb_np = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
            depth_np = np.load(depth_path)
            pose_w2c = np.load(pose_path)
            seg_map = np.load(seg_path)

            # Prepare tensors
            img_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
            img_tensor = transform_func(img_tensor).unsqueeze(0).to(device) # (1, 3, H, W)
            
            # Resize depth to feature map size for update function
            # But update function expects raw depth tensor to be interpolated?
            # Latent map uses:
            # depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()
            # depth_t_resized = F.interpolate(depth_t, (image_size, image_size), mode="nearest-exact")
            # depth_t = F.interpolate(depth_t_resized, (feat_h, feat_h), mode="nearest-exact").squeeze()
            
            depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()
            # Resize to image size first if needed, but update function handles extraction
            # Actually update function takes `depth_tensor` which determines output size of `get_3d_coordinates`.
            # We should pass depth resized to feature resolution.
            depth_t = F.interpolate(depth_t, (feature_map_size, feature_map_size), mode="nearest-exact").to(device)
            
            pose_tensor = torch.from_numpy(pose_w2c).float().to(device)
            
            # Update Map
            new_pts = update_latent_map_step(
                grid=grid,
                decoder=decoder,
                vision_model=vision_model,
                optimizer=optimizer,
                rgb_tensor=img_tensor,
                depth_tensor=depth_t,
                pose_world_to_cam=pose_tensor,
                seg_map=seg_map,
                excluded_ids=excluded_ids,
                intrinsics=intrinsics,
                original_image_size=original_image_size,
                device=device,
                scene_bounds=scene_bounds,
                num_iterations=update_steps,
            )
            
            if new_pts is not None:
                # Subsample new points to save memory if needed?
                # For now just append.
                P_background_list.append(new_pts)
                
            # Visualization Step
            if i % visualization_step == 0:
                print(f"[ONLINE] Visualizing step {i}...")
                
                # Consolidate background
                P_bg_curr = np.concatenate(P_background_list, axis=0)
                
                # Subsample for PCA if too large
                if P_bg_curr.shape[0] > max_points:
                    indices = np.random.choice(P_bg_curr.shape[0], max_points, replace=False)
                    P_bg_curr = P_bg_curr[indices]
                
                # Compute features for background
                # This queries the LATEST grid state
                features_bg = compute_features_for_points(
                    P_bg_curr, grid, decoder, device, batch_size=pca_batch_size
                )
                
                f_y_np = f_y.detach().cpu().numpy()
                
                # Combine features for PCA
                features_subset = np.concatenate([features_bg, f_y_np], axis=0)
                
                # PCA
                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(features_subset)
                scaler = MinMaxScaler(feature_range=(0, 1))
                pca_colors = scaler.fit_transform(pca_result)
                
                # Split colors back
                colors_bg = pca_colors[:features_bg.shape[0]]
                colors_Y = pca_colors[features_bg.shape[0]:]
                
                # Apply transform to Y
                R, t_vec = transforms_list[i]
                pts_Y_curr = pts_Y_init @ R.T + t_vec
                
                # Combine points and colors
                pts_all = np.vstack([P_bg_curr, pts_Y_curr])
                cols_all = np.vstack([colors_bg, colors_Y])
                
                pcd_dyn.points = o3d.utility.Vector3dVector(pts_all.astype(np.float64))
                pcd_dyn.colors = o3d.utility.Vector3dVector(cols_all.astype(np.float64))
                
                if len(frame_files_out) == 0:
                    vis.add_geometry(pcd_dyn)
                    if camera_extrinsic is not None:
                        ctr = vis.get_view_control()
                        params = ctr.convert_to_pinhole_camera_parameters()
                        params.extrinsic = camera_extrinsic
                        ctr.convert_from_pinhole_camera_parameters(params, False)
                else:
                    vis.update_geometry(pcd_dyn)
                
                vis.poll_events()
                vis.update_renderer()
                
                frame_path = os.path.join(temp_dir, f"frame_{len(frame_files_out):05d}.png")
                vis.capture_screen_image(frame_path, do_render=True)
                frame_files_out.append(frame_path)
        
        # Create video
        if frame_files_out:
            print(f"[ONLINE] Creating video from {len(frame_files_out)} frames...")
            output_dir = os.path.dirname(output_video_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with imageio.get_writer(output_video_path, fps=fps) as writer:
                for frame_file in frame_files_out:
                    writer.append_data(imageio.imread(frame_file))
            print(f"[ONLINE] Video saved to: {output_video_path}")
        else:
            print("[ONLINE][WARN] No frames were recorded.")

    finally:
        vis.destroy_window()
        shutil.rmtree(temp_dir)
        print(f"[ONLINE] Cleaned up temporary directory: {temp_dir}")


# ==============================================================================================
#  Main
# ==============================================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select one keypoint, backproject to 3D, find nearest latent point x, "
            "run 10cm ball query B(x), define Y (cos ≥ 0.7), store F(y) as f_y, "
            "track x with CoTracker, assume Y moves with x, and visualize dynamic PCA every 20 steps."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="DATASETS/behavior/processed_data/task-0021/episode_00210170/",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="head",
        choices=["head", "left_wrist", "right_wrist"],
        help="Name of the camera to use.",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=920,
        help="The starting frame index (0-based) used for keypoint selection.",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=2420,
        help="The ending frame index (exclusive).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for models: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default="trajectories_animation.html",
        help="(Optional) Output HTML file path for Plotly 3D visualization.",
    )
    parser.add_argument(
        "--pc_stride",
        type=int,
        default=8,
        help="Pixel stride for dense point cloud (for 3D trajectory + RGB pc).",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Process every N-th frame for tracking (e.g., 5 -> every 5th frame).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="mapping/config.yaml",
        help="Path to the config file for latent map."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the run directory containing trained models (grid.pt, decoder.pt) and point_cloud.npy."
    )
    parser.add_argument(
        "--ball_radius",
        type=float,
        default=0.3,
        help="Ball query radius (in meters). Default 0.10 (10 cm)."
    )
    parser.add_argument(
        "--cos_threshold",
        type=float,
        default=0.5,
        help="Cosine similarity threshold for 'high sim' visualization in B(x)."
    )
    parser.add_argument(
        "--dynamic_pca_interval",
        type=int,
        default=30,
        help="Do dynamic PCA visualization every this many time steps."
    )
    parser.add_argument(
        "--output_video_path",
        type=str,
        default="dynamic_pca_video.mp4",
        help="Path to save the dynamic PCA visualization video."
    )
    parser.add_argument(
        "--output_2d_video_path",
        type=str,
        default="cotracker_2d_visualization.mp4",
        help="Path to save the 2D tracks visualization video."
    )
    parser.add_argument(
        "--update_steps",
        type=int,
        default=1,
        help="Number of optimization steps per time-step."
    )
    args = parser.parse_args()

    # Placeholders for custom camera poses.
    # Press 'P' in an Open3D window to print the pose, then copy-paste it here.
    camera_extrinsic_dynamic_pca = np.array([[  0.73498117,   0.67314058,  -0.08175842, -20.85476113],
       [  0.04048914,  -0.16392308,  -0.98564185,   1.30857798],
       [ -0.67687762,   0.72111787,  -0.14773526,  12.67379289],
       [  0.        ,   0.        ,   0.        ,   1.        ]])

    # Device check
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    # ==========================================================================================
    #  Load Latent Map and Point Cloud
    # ==========================================================================================
    
    # 1. Load config
    # Priority: config.yaml in run_dir > args.config_path
    config_path = os.path.join(args.run_dir, "config.yaml")
    if os.path.exists(config_path):
        print(f"[INIT] Loading config from run directory: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"[INIT] Config not found in run_dir. Falling back to {args.config_path}")
        if not os.path.exists(args.config_path):
            print(f"[ERROR] Config file not found at {args.config_path}")
            return
            
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)

    # 2. Define model paths
    decoder_path = os.path.join(args.run_dir, "decoder.pt")
    
    grid_path = None
    point_cloud_path = None
    if os.path.exists(args.run_dir):
        # Find the first subdirectory containing a grid.pt
        for dir_name in sorted(os.listdir(args.run_dir)):
            env_dir = os.path.join(args.run_dir, dir_name)
            if os.path.isdir(env_dir):
                grid_path_candidate = os.path.join(env_dir, "grid.pt")
                pc_path_candidate = os.path.join(env_dir, "point_cloud.npy")

                if os.path.exists(grid_path_candidate) and os.path.exists(pc_path_candidate):
                    grid_path = grid_path_candidate
                    point_cloud_path = pc_path_candidate
                    print(f"[INIT] Found env '{dir_name}' with grid and point cloud.")
                    break
    
    if not all(p is not None and os.path.exists(p) for p in [decoder_path, grid_path, point_cloud_path]):
        print(f"[ERROR] Missing required files in run_dir '{args.run_dir}':")
        if not os.path.exists(decoder_path):
            print(f" - decoder.pt not found")
        if grid_path is None:
            print(f" - grid.pt not found in any env subdirectory")
        if point_cloud_path is None:
            print(f" - point_cloud.npy not found in any env subdirectory")
        return

    # 3. Load model configs
    if config['model_type'] == "clip":
        model_config = config['clip_model']
        image_size = model_config['image_size']
        patch_size = model_config['patch_size']
        feature_dim = model_config['feature_dim']
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ])
        
        clip_model_name  = model_config['name']
        clip_weights_id  = model_config['weights_id']
        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_weights_id
        )
        clip_model = clip_model.to(device).eval()
        vision_model = EvaClipWrapper(clip_model, output_dim=feature_dim).to(device).eval()
        print("[INIT] Loaded EVA-CLIP model.")

    elif config['model_type'] == "dino":
        model_config = config['dino_model']
        image_size = model_config['image_size']
        patch_size = model_config['patch_size']
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ])

        # Load DINOv3 backbone and initialize model
        WEIGHT_PATH = model_config['weights_path']
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dinov3'))
        if not os.path.exists(repo_dir):
             print(f"[WARN] dinov3 repo not found at {repo_dir}, trying 'dinov3' in current path or torch hub default")
             repo_dir = 'dinov3' # fall back

        print(f"[INIT] Loading DINOv3 from {repo_dir}")
        # Note: torch.hub.load with source='local' expects path to repo
        backbone = torch.hub.load(repo_dir, 'dinov3_vith16plus', source='local', weights=WEIGHT_PATH)
        vision_model = DINOv3Wrapper(backbone).to(device).eval()
        feature_dim = vision_model.feature_dim
        print(f"[INIT] Loaded DINOv3 model with feature dimension {feature_dim}.")
    else:
        raise ValueError(f"Unknown model type in config: {config['model_type']}")
    
    # Load Decoder
    print(f"[INIT] Loading decoder from {decoder_path}")
    decoder_config = config['decoder']
    grid_config = config['grid']
    decoder = ImplicitDecoder(
        voxel_feature_dim=grid_config['feature_dim'] * grid_config['levels'],
        hidden_dim=decoder_config['hidden_dim'],
        output_dim=feature_dim,
    ).to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval() # Frozen
    for param in decoder.parameters():
        param.requires_grad = False
    print("[INIT] Decoder loaded and frozen.")

    # Load Grid
    print(f"[INIT] Loading grid from {grid_path}")
    grid = VoxelHashTable(
        resolution=grid_config['resolution'],
        num_levels=grid_config['levels'],
        level_scale=grid_config['level_scale'],
        feature_dim=grid_config['feature_dim'],
        hash_table_size=grid_config['hash_table_size'],
        scene_bound_min=tuple(config['scene_min']),
        scene_bound_max=tuple(config['scene_max']),
        device=device,
    )
    grid.load_state_dict(torch.load(grid_path, map_location=device))
    grid.train() # Trainable
    print("[INIT] Grid loaded.")
    
    # Optimizer for Grid
    OPT_LR = config['training']['optimizer_lr']
    optimizer = torch.optim.Adam(grid.parameters(), lr=OPT_LR)
    print(f"[INIT] Optimizer initialized with LR={OPT_LR}")

    # 4. Load point cloud (latent map coordinates)
    print(f"[INIT] Loading point cloud from {point_cloud_path}")
    point_cloud = np.load(point_cloud_path)  # (N, 3) or (N, >=3)

    # ==========================================================================================
    #  Set up RGB/Depth/Pose paths & intrinsics
    # ==========================================================================================
    data_dir = os.path.join(args.data_dir, args.camera_name)
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    poses_dir = os.path.join(data_dir, "poses")
    seg_dir = os.path.join(data_dir, "seg_instance_id")
    intrinsics_path = os.path.join(data_dir, "intrinsics.txt")
    
    # Load instance ID mapping
    instance_id_json = os.path.join(args.data_dir, "instance_id_to_name.json")
    excluded_ids = get_excluded_ids(instance_id_json)
    print(f"[INIT] Excluded IDs (moving objects): {excluded_ids}")

    # Load intrinsics
    if not os.path.exists(intrinsics_path):
        print(f"[ERROR] Intrinsics file not found at {intrinsics_path}")
        return

    intrinsic_matrix = np.loadtxt(intrinsics_path)
    fx_orig = intrinsic_matrix[0, 0]
    fy_orig = intrinsic_matrix[1, 1]
    cx_orig = intrinsic_matrix[0, 2]
    cy_orig = intrinsic_matrix[1, 2]
    print(f"[INIT] Loaded intrinsics from {intrinsics_path}")

    # Load file lists (RGB, depth, pose, seg)
    try:
        rgb_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(".png")])
        depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(".npy")])
        pose_files = sorted([os.path.join(poses_dir, f) for f in os.listdir(poses_dir) if f.endswith(".npy")])
        seg_files = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith(".npy")])
    except FileNotFoundError as e:
        print(f"[ERROR] Data directory not found or incomplete. {e}")
        return

    if not (rgb_files and depth_files and pose_files and seg_files):
        print("[ERROR] One of rgb/depth/pose/seg is missing or empty.")
        return

    num_frames = len(rgb_files)
    # Ensure all lists have same length
    min_len = min(len(rgb_files), len(depth_files), len(pose_files), len(seg_files))
    if min_len < num_frames:
        print(f"[WARN] File counts differ. Truncating to {min_len} frames.")
        num_frames = min_len
        rgb_files = rgb_files[:num_frames]
        depth_files = depth_files[:num_frames]
        pose_files = pose_files[:num_frames]
        seg_files = seg_files[:num_frames]

    # Check start_frame / end_frame
    start = args.start_frame
    if start < 0 or start >= num_frames:
        print(f"[ERROR] Invalid start_frame {start}. It must be in [0, {num_frames-1}].")
        return

    if args.end_frame <= 0 or args.end_frame > num_frames:
        end_frame = num_frames
    else:
        end_frame = args.end_frame

    if start >= end_frame:
        print(f"[ERROR] start_frame ({start}) must be < end_frame ({end_frame}).")
        return

    print(f"[INFO] Using frame {start} for keypoint selection: {os.path.basename(rgb_files[start])}")
    print(f"[INFO] CoTracker frames will be from {start} to {end_frame - 1}, step={args.frame_step}.")

    # Visualize excluded masks before keypoint selection
    # print("[VIS] Visualizing excluded masks for the selected frame range...")
    # vis_rgb_files = rgb_files[start:end_frame:args.frame_step]
    # vis_seg_files = seg_files[start:end_frame:args.frame_step]
    # visualize_excluded_masks(
    #     rgb_files=vis_rgb_files,
    #     seg_files=vis_seg_files,
    #     excluded_ids=excluded_ids,
    #     output_path="mask_visualization.mp4",
    #     fps=10
    # )

    # ==========================================================================================
    #  Load start frame & select MULTIPLE keypoints (first one defines object)
    # ==========================================================================================
    first_rgb = imageio.imread(rgb_files[start])
    first_depth = np.load(depth_files[start])
    first_pose_world_to_cam = np.load(pose_files[start])

    h_rgb, w_rgb, _ = first_rgb.shape
    h_d, w_d = first_depth.shape
    
    # Update intrinsics if image size matches config vs original
    # The update function uses config['dataset']['original_image_size'] logic inside? 
    # No, we pass intrinsics. The vision wrapper resizes image to `image_size`.
    # The `get_3d_coordinates` needs intrinsics for the *input* depth map resolution.
    # The input depth is (h_d, w_d).
    # If h_d != 512 (original), we might need to scale intrinsics if they were for 512.
    # Assuming intrinsics.txt matches the loaded depth maps.
    
    keypoints_list = select_multiple_keypoints(first_rgb)
    if not keypoints_list:
        print("[ERROR] No keypoints selected. Exiting.")
        return

    # Use the first keypoint for the ball query / "object" definition
    u0, v0 = keypoints_list[0]
    print(f"[INFO] Selected {len(keypoints_list)} keypoints.")
    print(f"[INFO] Primary keypoint (for ball query) (u, v) = ({u0}, {v0})")

    # ==========================================================================================
    #  Backproject keypoint → 3D world coordinate at t=0
    # ==========================================================================================
    kp_world = compute_3d_point_from_depth(
        u=u0,
        v=v0,
        depth_map=first_depth,
        pose_world_to_cam=first_pose_world_to_cam,
        fx=fx_orig,
        fy=fy_orig,
        cx=cx_orig,
        cy=cy_orig,
    )

    if kp_world is None:
        print("[ERROR] Could not backproject primary keypoint to 3D (invalid depth?). Exiting.")
        return

    print(f"[3D] Primary keypoint world coordinate at t=0 = {kp_world}")

    # ==========================================================================================
    #  Ball query + latent feature cosine similarity + get Y, f_y
    # ==========================================================================================
    ball_res = ball_query_and_feature_similarity(
        keypoint_3d_world=kp_world,
        latent_points=point_cloud,
        grid=grid,
        decoder=decoder,
        device=device,
        radius=args.ball_radius,
        cos_threshold=args.cos_threshold,
    )
    Y_indices = ball_res["Y_indices"]
    f_y = ball_res["f_y"]

    if Y_indices.size == 0:
        print("[MAIN][WARN] Y is empty (no points with cos ≥ 0.7). Dynamic PCA will be skipped.")
    else:
        print(f"[MAIN] |Y| = {Y_indices.shape[0]}")

    # ==========================================================================================
    #  Build frames & run CoTracker to get x trajectory
    # ==========================================================================================
    frame_indices = list(range(start, end_frame, args.frame_step))
    if not frame_indices:
        print("[ERROR] No frames selected with the given start, end, and frame_step. Exiting.")
        return

    frames_list = []
    for idx in frame_indices:
        rgb = imageio.imread(rgb_files[idx])
        frames_list.append(rgb)

    frames_np = np.stack(frames_list, axis=0)  # (T, H, W, 3)
    T_frames = frames_np.shape[0]
    print(f"[INIT] Loaded {T_frames} RGB frames for CoTracker3.")

    # Subsample depth/pose lists according to frame_indices
    depth_files_sub = [depth_files[i] for i in frame_indices]
    pose_files_sub = [pose_files[i] for i in frame_indices]

    # Run CoTracker on ALL selected keypoints
    print(f"[TRACK] Running CoTracker3 online tracking for {len(keypoints_list)} keypoints...")
    tracks_xy, visibilities = run_cotracker_online_multi(
        frames_np,
        keypoints=keypoints_list,
        device=device,
    )
    if tracks_xy.shape[0] != T_frames:
        print(
            f"[WARN] CoTracker returned {tracks_xy.shape[0]} frames but input has {T_frames}. "
            "Will use the min length."
        )

    # Compute 3D trajectory for ALL keypoints
    positions_3d, colors_rgb, pc_positions, pc_colors = compute_3d_positions_and_colors_over_time(
        frames_np=frames_np,
        tracks_xy=tracks_xy,
        visibilities=visibilities,
        depth_files=depth_files_sub,
        pose_files=pose_files_sub,
        start_frame_index=0,  # because we already subsampled lists
        fx=fx_orig,
        fy=fy_orig,
        cx=cx_orig,
        cy=cy_orig,
        vis_threshold=0.8,
        point_stride=args.pc_stride,
    )
    
    # positions_3d: (T_eff, N, 3)
    # Estimate rigid transforms from ALL keypoint trajectories
    transforms_list = estimate_transforms_over_time(positions_3d)

    # ==========================================================================================
    #  Online Update + Dynamic PCA visualization
    # ==========================================================================================
    if Y_indices.size > 0:
        run_online_update_and_visualize(
            point_cloud_init=point_cloud,
            grid=grid,
            decoder=decoder,
            vision_model=vision_model,
            optimizer=optimizer,
            rgb_files=rgb_files,
            depth_files=depth_files,
            pose_files=pose_files,
            seg_files=seg_files,
            frame_indices=frame_indices,
            transforms_list=transforms_list,
            Y_indices=Y_indices,
            f_y=f_y,
            excluded_ids=excluded_ids,
            intrinsics=(fx_orig, fy_orig, cx_orig, cy_orig),
            original_image_size=config['dataset']['original_image_size'],
            scene_bounds=(tuple(config['scene_min']), tuple(config['scene_max'])),
            device=device,
            transform_func=transform,
            feature_map_size=image_size // patch_size,
            visualization_step=args.dynamic_pca_interval,
            output_video_path=args.output_video_path,
            fps=10,
            camera_extrinsic=camera_extrinsic_dynamic_pca,
            update_steps=args.update_steps,
        )
    else:
        print("[MAIN] Skipping dynamic PCA because Y is empty.")


if __name__ == "__main__":
    main()
