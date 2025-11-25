import numpy as np
import os
import imageio.v2 as imageio
import argparse
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
import open3d as o3d

import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import tempfile
import shutil
import copy


# local co-tracker repo (for imports etc. if needed)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "co-tracker"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mapping.mapping_lib.implicit_decoder import ImplicitDecoder
from mapping.mapping_lib.voxel_hash_table import VoxelHashTable


class SparseLevel:
    """
    A single level of sparse voxels.
    Stores features in Tensors:
      - coords: (M, 3) LongTensor
      - features: (M, d) FloatTensor
    Uses a CPU dictionary for coordinate -> index mapping (fast lookup).
    """
    def __init__(self, res, smin, corner_offsets, feature_dim, device):
        self.res = res
        self.smin = smin
        self.corner_offsets = corner_offsets
        self.d = feature_dim
        self.device = device
        
        # Storage
        self.coords = None    # (M, 3)
        self.features = None  # (M, d)
        self.coord_to_idx = {} # Dict: (x,y,z) -> index in self.features

    def add_features(self, grid_coords: torch.Tensor, features: torch.Tensor):
        """
        grid_coords: (M, 3) long
        features: (M, d) float
        """
        self.coords = grid_coords.to(self.device)
        self.features = features.to(self.device)
        
        # Build look-up table on CPU for speed
        # (Since M is relatively small for a single object, this is efficient)
        coords_np = grid_coords.cpu().numpy()
        for i in range(coords_np.shape[0]):
            key = tuple(coords_np[i])
            self.coord_to_idx[key] = i

    def query(self, pts: torch.Tensor):
        """
        Trilinear interpolation.
        pts: (N, 3)
        """
        N = pts.shape[0]
        
        # 1. Compute coordinates and weights
        q = (pts - self.smin) / self.res
        base = torch.floor(q).long()
        
        # (N, 8, 3)
        corner_coords = base[:, None, :] + self.corner_offsets[None, :, :]
        corner_coords_flat = corner_coords.view(-1, 3).cpu().numpy() # (N*8, 3)
        
        # 2. Lookup indices
        # We need to find which of these 8*N corners exist in our sparse grid.
        # Map coord -> feature index. -1 if not found.
        indices = np.full(N * 8, -1, dtype=np.int64)
        
        # Speed optimization: Only iterate corners, lookup in dict
        for i in range(N * 8):
            key = tuple(corner_coords_flat[i])
            if key in self.coord_to_idx:
                indices[i] = self.coord_to_idx[key]
                
        # 3. Gather features
        indices_torch = torch.from_numpy(indices).to(self.device)
        
        # Mask for valid indices
        mask = indices_torch >= 0
        
        # Output buffer (N*8, d)
        flat_features = torch.zeros(N * 8, self.d, device=self.device)
        
        if mask.any():
            # Gather valid features from the big tensor
            valid_indices = indices_torch[mask]
            flat_features[mask] = self.features[valid_indices]
            
        # Reshape to (N, 8, d)
        corner_features = flat_features.view(N, 8, self.d)
        
        # 4. Interpolate
        frac = q - base.float()
        wx = torch.stack([1 - frac[:, 0], frac[:, 0]], 1)
        wy = torch.stack([1 - frac[:, 1], frac[:, 1]], 1)
        wz = torch.stack([1 - frac[:, 2], frac[:, 2]], 1)
        
        w = wx[:, [0, 1, 0, 1, 0, 1, 0, 1]] * \
            wy[:, [0, 0, 1, 1, 0, 0, 1, 1]] * \
            wz[:, [0, 0, 0, 0, 1, 1, 1, 1]]
            
        return (corner_features * w.unsqueeze(-1)).sum(1)


class ObjectGrid:
    """
    A structure that holds ONLY the relevant voxels (coordinates + features).
    Mimics VoxelHashTable's interface for querying.
    NOW supports rigid body pose (R, t). Queries are transformed to local frame.
    """
    def __init__(self, original_grid, points_of_interest, device):
        self.levels = []
        self.device = device
        
        # Pose: World -> Grid (Canonical)
        # p_world = R @ p_local + t
        # p_local = R.T @ (p_world - t)
        self.R = torch.eye(3, device=device)
        self.t = torch.zeros(3, device=device)
        
        # Extract features for each level
        for lv_idx, lv_src in enumerate(original_grid.levels):
            # Create sparse level
            sparse_lv = SparseLevel(
                lv_src.res, 
                lv_src.smin, 
                lv_src.corner_offsets, 
                lv_src.d, 
                device
            )
            
            # 1. Find all voxel coords accessed by points
            # We duplicate the logic from _TrainLevel.query to get 'idx' (grid coords)
            with torch.no_grad():
                pts = points_of_interest.float().to(device)
                q_detached = (pts - lv_src.smin) / lv_src.res
                base = torch.floor(q_detached).long()
                # (N, 8, 3)
                corner_coords = base[:, None, :] + lv_src.corner_offsets[None, :, :]
                unique_coords = corner_coords.view(-1, 3).unique(dim=0) # (M, 3)
            
            # 2. Retrieve features from original grid using these coords
            # We need to hash them to find the index in voxel_features
            vid = (unique_coords * lv_src.primes).sum(-1) % lv_src.buckets
            features = lv_src.voxel_features[vid] # (M, d)
            
            # 3. Add to sparse level
            if unique_coords.shape[0] > 0:
                sparse_lv.add_features(unique_coords, features)
            self.levels.append(sparse_lv)
            
            print(f"[SparseGrid] Level {lv_idx}: stored {unique_coords.shape[0]} voxels.")

    def update_pose(self, R: np.ndarray, t: np.ndarray):
        """
        Update the grid's pose in the world.
        R: (3, 3) numpy array
        t: (3,) numpy array
        """
        self.R = torch.from_numpy(R).float().to(self.device)
        self.t = torch.from_numpy(t).float().to(self.device)

    def query_voxel_feature(self, pts, mark_accessed=False):
        # pts: (N, 3) in WORLD coordinates
        
        # Transform World -> Local (Canonical)
        # p_local = (p_world - t) @ R
        # Note: usually p_world = R @ p_local + t
        # So p_world - t = R @ p_local
        # R.T @ (p_world - t) = p_local
        
        # pts is (N, 3), so we do (pts - t) @ R 
        # (equivalent to (R.T @ (pts-t).T).T )
        
        pts_local = torch.matmul(pts - self.t, self.R)
        
        # mark_accessed is ignored for sparse grid
        feats = []
        for lv in self.levels:
            feats.append(lv.query(pts_local))
        return torch.cat(feats, -1)


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
        print(f"\\n[{window_name}] Camera Extrinsic Pose (copy-paste this):")
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


def visualize_se3_trajectory(
    transforms: list,
    centroid: np.ndarray = None,
    step: int = 5,
    window_name: str = "SE(3) Trajectory"
):
    """
    Visualizes the estimated SE(3) trajectory using Open3D coordinate frames.
    
    Args:
        transforms: List of (R, t) tuples.
        centroid: Optional (3,) array. If provided, frames are visualized at R @ centroid + t.
                  If None, frames are visualized at t (trajectory of the origin).
        step: Visualize every step-th frame to reduce clutter.
    """
    if centroid is None:
        centroid = np.zeros(3)
        
    geometries = []
    traj_points = []
    
    print(f"[VIS] Generating SE(3) visualization for {len(transforms)} steps (stride={step})...")
    
    for i in range(0, len(transforms), step):
        R, t = transforms[i]
        
        # Compute position for the frame origin
        # If centroid is provided, this is the transformed centroid position
        pos = R @ centroid + t
        traj_points.append(pos)
        
        # Coordinate frame
        # size=0.1 meters (10cm)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Apply pose: first rotate, then translate to 'pos'
        # Note: create_coordinate_frame makes a frame at (0,0,0).
        # We want its origin to be at 'pos', and orientation 'R'.
        T_mat = np.eye(4)
        T_mat[:3, :3] = R
        T_mat[:3, 3] = pos
        
        mesh_frame.transform(T_mat)
        geometries.append(mesh_frame)

    # Connect points with a line
    if len(traj_points) > 1:
        lines = [[i, i + 1] for i in range(len(traj_points) - 1)]
        colors = [[1, 0, 0] for _ in range(len(lines))]  # Red trajectory line
        
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(traj_points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)
        
    print(f"[VIS] Displaying {window_name}. Close window to continue...")
    draw_geometries_with_key_callbacks(geometries, window_name=window_name)


# ==============================================================================================
#  Plotly 3D animation (kept for reference)
# ==============================================================================================

def _time_to_color(t: int, T: int) -> str:
    """
    Simple blue -> red gradient over time.
    t in [0, T-1]
    """
    if T <= 1:
        return "rgb(0,0,255)"
    ratio = t / float(T - 1)
    r = int(255 * ratio)
    g = 0
    b = int(255 * (1.0 - ratio))
    return f"rgb({r},{g},{b})"


def visualize_animation_plotly(
    kp_positions_3d: np.ndarray,   # (T, N, 3) with NaN in invalid
    kp_colors_rgb: np.ndarray,     # (T, N, 3) uint8
    pc_positions: list[np.ndarray],  # len T, each (M_t, 3)
    pc_colors: list[np.ndarray],     # len T, each (M_t, 3)
    output_html: str,
):
    """
    Not used in the dynamic PCA part, kept for reference.
    """
    T, N, _ = kp_positions_3d.shape

    # Initial frame
    t0 = 0
    pts0_pc = pc_positions[t0]   # (M0, 3)
    cols0_pc = pc_colors[t0]     # (M0, 3)
    pc_color_strings0 = [
        f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in cols0_pc
    ]

    data = []

    # dense point cloud at t0
    if pts0_pc.shape[0] > 0:
        data.append(
            go.Scatter3d(
                x=pts0_pc[:, 0],
                y=pts0_pc[:, 1],
                z=pts0_pc[:, 2],
                mode="markers",
                marker=dict(size=2, color=pc_color_strings0),
                name="points_t",
            )
        )
    else:
        data.append(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=2, color="rgb(150,150,150)"),
                name="points_t",
            )
        )

    # trajectories at t0
    for n in range(N):
        pts = kp_positions_3d[: t0 + 1, n, :]
        mask = ~np.isnan(pts[:, 0])
        pts_valid = pts[mask]
        if pts_valid.shape[0] > 0:
            x_n = pts_valid[:, 0]
            y_n = pts_valid[:, 1]
            z_n = pts_valid[:, 2]
            times = np.arange(0, t0 + 1)[mask]
        else:
            x_n = y_n = z_n = []
            times = []

        data.append(
            go.Scatter3d(
                x=x_n,
                y=y_n,
                z=z_n,
                mode="lines+markers",
                line=dict(width=3, color="rgba(220,220,220,0.8)"),
                marker=dict(
                    size=4,
                    color=times,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=T - 1,
                    showscale=True if n == 0 and t0 == 0 else False,
                ),
                name=f"keypoint_{n}",
                showlegend=True if n == 0 else False,
            )
        )

    frames = []
    for t in range(T):
        pts_t_pc = pc_positions[t]
        cols_t_pc = pc_colors[t]
        pc_color_strings_t = [
            f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in cols_t_pc
        ]

        frame_data = []

        # dense pc
        if pts_t_pc.shape[0] > 0:
            frame_data.append(
                go.Scatter3d(
                    x=pts_t_pc[:, 0],
                    y=pts_t_pc[:, 1],
                    z=pts_t_pc[:, 2],
                    mode="markers",
                    marker=dict(size=2, color=pc_color_strings_t),
                    name="points_t",
                )
            )
        else:
            frame_data.append(
                go.Scatter3d(
                    x=[],
                    y=[],
                    z=[],
                    mode="markers",
                    marker=dict(size=2, color="rgb(150,150,150)"),
                    name="points_t",
                )
            )

        # trajectories up to t
        for n in range(N):
            pts = kp_positions_3d[: t + 1, n, :]
            mask = ~np.isnan(pts[:, 0])
            pts_valid = pts[mask]
            if pts_valid.shape[0] > 0:
                x_n = pts_valid[:, 0]
                y_n = pts_valid[:, 1]
                z_n = pts_valid[:, 2]
                times = np.arange(0, t + 1)[mask]
            else:
                x_n = y_n = z_n = []
                times = []

            frame_data.append(
                go.Scatter3d(
                    x=x_n,
                    y=y_n,
                    z=z_n,
                    mode="lines+markers",
                    line=dict(width=3, color="rgba(220,220,220,0.8)"),
                    marker=dict(
                        size=4,
                        color=times,
                        colorscale="Viridis",
                        cmin=0,
                        cmax=T - 1,
                        showscale=False,
                    ),
                    name=f"keypoint_{n}",
                    showlegend=False,
                )
            )

        frames.append(go.Frame(data=frame_data, name=str(t)))

    sliders = [
        {
            "steps": [
                {
                    "args": [[str(t)], {"frame": {"duration": 50, "redraw": True},
                                        "mode": "immediate"}],
                    "label": str(t),
                    "method": "animate",
                }
                for t in range(T)
            ],
            "transition": {"duration": 0},
            "x": 0.05,
            "y": 0,
            "currentvalue": {"font": {"size": 14}, "prefix": "t = ", "visible": True, "xanchor": "right"},
            "len": 0.9,
        }
    ]

    updatemenus = [
        {
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 50, "redraw": True},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                            "mode": "immediate",
                        },
                    ],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                        },
                    ],
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": True,
            "x": 0.1,
            "y": 0,
            "xanchor": "right",
            "yanchor": "top",
        }
    ]

    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        title="3D Keypoint Trajectories with Time-Colored Points + Dense RGB Point Cloud",
        scene=dict(
            xaxis_title="X (world)",
            yaxis_title="Y (world)",
            zaxis_title="Z (world)",
            aspectmode="data",
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        sliders=sliders,
        updatemenus=updatemenus,
    )

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    pio.write_html(fig, output_html, auto_open=False)
    print(f"[PLOT] Saved Plotly 3D animation to: {output_html}")


# ==============================================================================================
#  PCA Visualization (latent feature -> color) for static case
# ==============================================================================================

def visualize_pca_open3d(
    points_3d: np.ndarray,
    grid: VoxelHashTable,
    decoder: ImplicitDecoder,
    device: str,
    camera_extrinsic: np.ndarray = None,
):
    """
    Visualizes a point cloud with colors derived from PCA on its latent features using Open3D.
    """
    # --- Sampling ---
    max_points = 2000000
    if points_3d.shape[0] > max_points:
        print(f"[PCA] Sampling {max_points} points from the original {points_3d.shape[0]} points.")
        indices = np.random.choice(points_3d.shape[0], max_points, replace=False)
        points_3d = points_3d[indices]

    print(f"[PCA] Visualizing PCA of features for {points_3d.shape[0]} points.")
    
    # Process in batches to avoid OOM
    batch_size = 100000
    all_features = []

    with torch.no_grad():
        for i in range(0, points_3d.shape[0], batch_size):
            batch_points = points_3d[i:i+batch_size]
            points_tensor = torch.from_numpy(batch_points).float().to(device)
            
            voxel_feat = grid.query_voxel_feature(points_tensor, mark_accessed=False)
            pred_feat = decoder(voxel_feat)
            all_features.append(pred_feat.cpu().numpy())

    features_np = np.concatenate(all_features, axis=0)

    print("[PCA] Running PCA on features...")
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_np)

    scaler = MinMaxScaler(feature_range=(0.1, 1))
    pca_colors = scaler.fit_transform(pca_result)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(pca_colors)

    print("[PCA] Displaying point cloud with Open3D...")
    draw_geometries_with_key_callbacks(
        [pcd],
        window_name="PCA of Latent Features",
        camera_extrinsic=camera_extrinsic,
    )


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
#  NEW: Ball query + latent feature cosine similarity visualization + Y, f_y 반환
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
    # draw_geometries_with_key_callbacks(
    #     [pcd_ball],
    #     window_name=f"Ball query (r={radius} m) & cosine ≥ {cos_threshold}",
    # )

    return {
        "x": x,
        "x_idx": x_idx,
        "idx_ball": idx_ball,
        "points_ball": points_ball,
        "sims": sims,
        "Y_indices": Y_indices_global,
        "f_y": f_y,
    }


def distill_object_features(
    grid: VoxelHashTable,
    object_grid: ObjectGrid,
    decoder: ImplicitDecoder,
    points_Y_local: np.ndarray, # (Ny, 3)
    R_final: np.ndarray,
    t_final: np.ndarray,
    device: str,
    lr: float = 0.01,
    num_steps: int = 500,
):
    """
    Distills features from object_grid (at final pose) to the global grid (at final world coords).
    """
    print(f"[DISTILL] Starting distillation to canonical grid... (steps={num_steps}, lr={lr})")
    
    # 1. Prepare target features (from ObjectGrid)
    # Update pose to final
    object_grid.update_pose(R_final, t_final)
    
    # Compute final world coordinates
    # p_world = p_local @ R^T + t
    points_Y_world = points_Y_local @ R_final.T + t_final
    
    # Get Target Features (Frozen)
    with torch.no_grad():
        pts_tensor = torch.from_numpy(points_Y_world).float().to(device)
        # This queries ObjectGrid, which transforms world->local and gets stored features
        target_voxel_feats = object_grid.query_voxel_feature(pts_tensor)
        target_latents = decoder(target_voxel_feats) # (Ny, C)
        target_latents = F.normalize(target_latents, dim=-1).detach()

    # 2. Optimize Global Grid
    # We want global_grid(points_Y_world) -> similar to target_latents
    
    # Enable gradients for grid
    grid.train()
    # Note: ImplicitDecoder is usually frozen here
    
    # Explicitly freeze decoder parameters to be safe
    for param in decoder.parameters():
        param.requires_grad = False
    decoder.eval()

    optimizer = optim.Adam(grid.parameters(), lr=lr)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Query global grid at NEW positions
        global_voxel_feats = grid.query_voxel_feature(pts_tensor, mark_accessed=True) 
        
        pred_latents = decoder(global_voxel_feats)
        pred_latents_norm = F.normalize(pred_latents, dim=-1)
        
        # Cosine Similarity Loss: 1 - cos_sim
        # maximize sum(a*b)
        cos_sim = (pred_latents_norm * target_latents).sum(dim=-1).mean()
        loss = 1.0 - cos_sim
        
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"[DISTILL] Step {step}: Loss = {loss.item():.6f}, CosSim = {cos_sim.item():.6f}")

    print(f"[DISTILL] Finished. Final Loss = {loss.item():.6f}")
    grid.eval()


# ==============================================================================================
#  NEW: Dynamic PCA visualization every K time steps
# ==============================================================================================

def visualize_dynamic_pca_open3d(
    points_all: np.ndarray,        # P, (N, ≥3), world coords used in latent map
    grid: VoxelHashTable,
    object_grid: ObjectGrid,  # CHANGED: Now accepts ObjectGrid
    decoder: ImplicitDecoder,
    device: str,
    Y_indices: np.ndarray,         # indices of Y in P
    transforms: list,              # List of (R, t) tuples, one per time step
    step_interval: int = 20,
    max_points: int = 200000,
    batch_size: int = 100000,
    camera_extrinsic: np.ndarray = None,
    output_video_path: str = "dynamic_pca.mp4",
    fps: int = 30,
):
    """
    Dynamic PCA visualization that runs non-blockingly and saves to video.
      - P: The entire point cloud.
      - Y: A subset of P (by indices). Voxels for Y are in object_grid.
      - P\Y: Features for these points are always queried from global grid.
      - transforms: List of (R, t) such that p_t = R @ p_0 + t.
      - Assumption: All y in Y move according to the rigid transform.
    """
    if points_all.shape[0] == 0:
        print("[DYN-PCA][ERROR] Empty point cloud P.")
        return

    # Ensure we only use XYZ from P
    if points_all.shape[1] > 3:
        P_xyz = points_all[:, :3]
    else:
        P_xyz = points_all

    N = P_xyz.shape[0]

    Y_indices = np.unique(Y_indices.astype(np.int64))
    Ny = Y_indices.shape[0]
    if Ny == 0:
        print("[DYN-PCA][WARN] Y is empty (no points with cos ≥ 0.7). Nothing to visualize specially.")
        return

    print(f"[DYN-PCA] Total points in P: {N}, |Y|: {Ny}")

    # Build sampling for P\Y, ensuring all Y are always included
    all_indices = np.arange(N, dtype=np.int64)
    mask_y = np.zeros(N, dtype=bool)
    mask_y[Y_indices] = True
    others_idx = all_indices[~mask_y]
    N_others = others_idx.shape[0]

    # We always include all Y. Others are randomly subsampled up to max_points - Ny.
    max_points_effective = max(max_points, Ny)
    max_points_effective = min(max_points_effective, N)  # cannot exceed N
    max_others = max_points_effective - Ny
    if max_others < 0:
        max_others = 0

    if N_others > max_others and max_others > 0:
        others_sampled = np.random.choice(others_idx, max_others, replace=False)
    else:
        others_sampled = others_idx

    N_others_sampled = others_sampled.shape[0]
    print(f"[DYN-PCA] Sampling {N_others_sampled} points from P\\Y and keeping all {Ny} points in Y.")

    # Pre-separate subsets
    points_others_xyz0 = P_xyz[others_sampled]    # (N_others_sampled, 3)
    points_Y_xyz0 = P_xyz[Y_indices]             # (Ny, 3)

    # Features:
    #  - P\Y sampled: query GLOBAL latent map
    #  - Y: query OBJECT latent map (at p0)
    
    # 1. Others
    if N_others_sampled > 0:
        features_others = compute_features_for_points(
            points_others_xyz0,
            grid,
            decoder,
            device,
            batch_size=batch_size,
        )
    else:
        features_others = np.zeros((0, decoder.output_dim), dtype=np.float32)

    # 2. Y - Query object_grid
    # Initialize object_grid pose to Identity (t=0)
    object_grid.update_pose(np.eye(3), np.zeros(3))
    
    # Now we query with WORLD coordinates (which are p0 here), 
    # and object_grid will transform them to local (p0) internally.
    features_Y = compute_features_for_points(
        points_Y_xyz0,
        object_grid,
        decoder,
        device,
        batch_size=batch_size,
    )

    features_subset = np.concatenate([features_others, features_Y], axis=0)  # (M, C)
    print(f"[DYN-PCA] Computing PCA on {features_subset.shape[0]} feature vectors.")

    # PCA(fixed), colors fixed over time (features are time-invariant)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_subset)

    scaler = MinMaxScaler(feature_range=(0.1, 1))
    pca_colors = scaler.fit_transform(pca_result)  # (M, 3)

    # Transforms check
    T = len(transforms)
    if T == 0:
        print("[DYN-PCA][ERROR] Empty transforms list.")
        return
    
    t0 = 0
    print(f"[DYN-PCA] Using t0 = {t0} as reference frame (assuming R=I, t=0).")

    # Pre-separate subsets for convenience
    points_others_xyz0 = P_xyz[others_sampled]    # (N_others_sampled, 3)
    points_Y_xyz0 = P_xyz[Y_indices]             # (Ny, 3)

    # --- Initial frame visualization (blocking) ---
    # print(f"\n[DYN-PCA] Displaying initial frame at t={t0}. Close the window to proceed.")
    
    # t=0 transform should be identity, but let's apply transforms[0] just in case
    R0, t_vec0 = transforms[t0]
    
    # Static background (others) - assumes they don't move or we only move Y?
    # The requirement is "selected particles move", so others are static?
    # Original code kept others static: pts_others_t = points_others_xyz0
    pts_others_t0 = points_others_xyz0

    # Apply transform to Y
    # points_Y_xyz0 is (Ny, 3)
    pts_Y_t0 = points_Y_xyz0 @ R0.T + t_vec0

    pts_t0 = np.vstack([pts_others_t0, pts_Y_t0])

    pcd_initial = o3d.geometry.PointCloud()
    pcd_initial.points = o3d.utility.Vector3dVector(pts_t0.astype(np.float64))
    pcd_initial.colors = o3d.utility.Vector3dVector(pca_colors.astype(np.float64))

    # draw_geometries_with_key_callbacks(
    #     [pcd_initial],
    #     window_name=f"Dynamic PCA (t={t0}) - Close window to continue",
    #     camera_extrinsic=camera_extrinsic,
    # )
    
    # input("\nPress Enter to start video recording...")
    # --- End of initial frame visualization ---

    # time steps to visualize
    time_steps = list(range(t0, T, step_interval))
    print(f"[DYN-PCA] Visualizing and recording {len(time_steps)} time steps...")

    # Non-blocking visualization and recording
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Dynamic PCA of Latent Features (Recording)")
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    pcd_dyn = o3d.geometry.PointCloud()

    try:
        for i, t in enumerate(time_steps):
            if t >= T:
                continue

            R, t_vec = transforms[t]

            # Others stay static
            pts_others_t = points_others_xyz0
            
            # Y moves with rigid transform
            pts_Y_t = points_Y_xyz0 @ R.T + t_vec

            pts_t = np.vstack([pts_others_t, pts_Y_t])

            # NEW: Update object_grid pose so it follows the object
            object_grid.update_pose(R, t_vec)

            pcd_dyn.points = o3d.utility.Vector3dVector(pts_t.astype(np.float64))
            # Recalculate features/colors if we wanted dynamic colors, 
            # but here we use fixed PCA colors based on initial features.
            # Wait, if we want to prove the grid is moving, we should technically 
            # re-query features at pts_Y_t using the updated object_grid?
            # The current PCA colors are fixed from t=0. 
            # To strictly follow "object_grid is moving", let's re-verify:
            # - PCA colors were computed at t=0.
            # - If we re-computed features at t using pts_Y_t and updated object_grid,
            #   we should get the SAME features (and thus same colors).
            # - So reusing pca_colors is valid and efficient.
            
            pcd_dyn.colors = o3d.utility.Vector3dVector(pca_colors.astype(np.float64))

            if i == 0:
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

            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            vis.capture_screen_image(frame_path, do_render=True)
            frame_files.append(frame_path)

        # Create video from frames
        if frame_files:
            print(f"[DYN-PCA] Creating video from {len(frame_files)} frames...")
            output_dir = os.path.dirname(output_video_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with imageio.get_writer(output_video_path, fps=fps) as writer:
                for frame_file in frame_files:
                    writer.append_data(imageio.imread(frame_file))
            print(f"[DYN-PCA] Video saved to: {output_video_path}")
        else:
            print("[DYN-PCA][WARN] No frames were recorded.")

    finally:
        vis.destroy_window()
        shutil.rmtree(temp_dir)
        print(f"[DYN-PCA] Cleaned up temporary directory: {temp_dir}")


# ==============================================================================================
#  2D Tracks Visualization
# ==============================================================================================

def visualize_2d_tracks_video(
    frames_np: np.ndarray,
    tracks_xy: np.ndarray,
    visibilities: np.ndarray,
    output_path: str,
    fps: int = 10,
):
    """
    Visualizes 2D tracks on the video frames and saves as a video.
    """
    print(f"[VIS] Saving 2D tracks video to {output_path}...")
    T, N, _ = tracks_xy.shape
    if T == 0:
        return

    # Generate random colors for each keypoint (RGB)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(N, 3), dtype=np.uint8)

    writer = imageio.get_writer(output_path, fps=fps)

    for t in range(T):
        img = frames_np[t].copy()
        
        for n in range(N):
            vis = visibilities[t, n]
            if vis > 0.5:
                x, y = tracks_xy[t, n]
                if np.isnan(x) or np.isnan(y):
                    continue
                
                color = (int(colors[n, 0]), int(colors[n, 1]), int(colors[n, 2]))
                
                # Draw point
                cv2.circle(img, (int(x), int(y)), 4, color, -1)
        
        writer.append_data(img)

    writer.close()
    print(f"[VIS] Saved 2D tracks video to {output_path}")


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
        default=3,
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
        default=1,
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
        feature_dim = config['clip_model']['feature_dim']
    elif config['model_type'] == "dino":
        # For DINOv3 ViT-H/16plus, the feature dimension is 1280.
        feature_dim = 1280
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
    decoder.eval()
    print("[INIT] Decoder loaded.")

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
    grid.eval()
    print("[INIT] Grid loaded.")

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
    intrinsics_path = os.path.join(data_dir, "intrinsics.txt")

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

    # Load file lists (RGB, depth, pose)
    try:
        rgb_files = sorted(
            [
                os.path.join(rgb_dir, f)
                for f in os.listdir(rgb_dir)
                if f.endswith(".png")
            ]
        )
        depth_files = sorted(
            [
                os.path.join(depth_dir, f)
                for f in os.listdir(depth_dir)
                if f.endswith(".npy")
            ]
        )
        pose_files = sorted(
            [
                os.path.join(poses_dir, f)
                for f in os.listdir(poses_dir)
                if f.endswith(".npy")
            ]
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Data directory not found or incomplete. {e}")
        return

    if not (rgb_files and depth_files and pose_files):
        print("[ERROR] One of rgb/depth/pose is missing or empty.")
        return

    num_frames = len(rgb_files)
    if not (len(depth_files) == num_frames and len(pose_files) == num_frames):
        print(
            f"[WARN] Number of frames differ: rgb={len(rgb_files)}, "
            f"depth={len(depth_files)}, pose={len(pose_files)}"
        )
        num_frames = min(len(rgb_files), len(depth_files), len(pose_files))
        rgb_files = rgb_files[:num_frames]
        depth_files = depth_files[:num_frames]
        pose_files = pose_files[:num_frames]

    # Check start_frame / end_frame
    start = args.start_frame
    if start < 0 or start >= num_frames:
        print(
            f"[ERROR] Invalid start_frame {start}. It must be in [0, {num_frames-1}]."
        )
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

    # ==========================================================================================
    #  Load start frame & select MULTIPLE keypoints (first one defines object)
    # ==========================================================================================
    first_rgb = imageio.imread(rgb_files[start])
    first_depth = np.load(depth_files[start])
    first_pose_world_to_cam = np.load(pose_files[start])

    h_rgb, w_rgb, _ = first_rgb.shape
    h_d, w_d = first_depth.shape
    if (h_rgb, w_rgb) != (h_d, w_d):
        print(
            f"[WARN] RGB size {first_rgb.shape} and depth size {first_depth.shape} differ. "
            "Assuming intrinsics correspond to depth resolution. "
            "If not, you may need resizing / scaling."
        )

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

    # Visualize 2D tracks
    # visualize_2d_tracks_video(
    #     frames_np=frames_np,
    #     tracks_xy=tracks_xy,
    #     visibilities=visibilities,
    #     output_path=args.output_2d_video_path,
    #     fps=10,
    # )

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
    transforms = estimate_transforms_over_time(positions_3d)

    # ==========================================================================================
    #  Visualize SE(3) Trajectory (Coordinate Frames)
    # ==========================================================================================
    # pts_0 = positions_3d[0]
    # valid_0 = ~np.isnan(pts_0[:, 0])
    # if np.any(valid_0):
    #     centroid_0 = np.mean(pts_0[valid_0], axis=0)
    #     # Visualize every 5th frame to avoid clutter
    #     visualize_se3_trajectory(transforms, centroid=centroid_0, step=5)
    # else:
    #     print("[WARN] Cannot visualize SE(3) trajectory: no valid points at t=0.")

    # ==========================================================================================
    #  Visualize Keypoint Trajectories + Point Cloud (Plotly HTML)
    # ==========================================================================================
    # print(f"[PLOT] Generating Plotly HTML animation: {args.output_html}")
    # visualize_animation_plotly(
    #     kp_positions_3d=positions_3d,
    #     kp_colors_rgb=colors_rgb,
    #     pc_positions=pc_positions,
    #     pc_colors=pc_colors,
    #     output_html=args.output_html,
    # )

    # ==========================================================================================
    #  Dynamic PCA visualization every 20 steps
    # ==========================================================================================
    if Y_indices.size > 0:
        # NEW: Create ObjectGrid holding ONLY object voxels
        print("[MAIN] Creating ObjectGrid with only object voxels...")
        
        points_Y = point_cloud[Y_indices]
        if points_Y.shape[1] > 3: 
            points_Y = points_Y[:, :3]
            
        points_Y_tensor = torch.from_numpy(points_Y).float() # .to(device) done inside init
        
        object_grid = ObjectGrid(grid, points_Y_tensor, device)

        # visualize_dynamic_pca_open3d(
        #     points_all=point_cloud,
        #     grid=grid,
        #     object_grid=object_grid,
        #     decoder=decoder,
        #     device=device,
        #     Y_indices=Y_indices,
        #     transforms=transforms,
        #     step_interval=args.dynamic_pca_interval,
        #     max_points=2000000,
        #     batch_size=100000,
        #     camera_extrinsic=camera_extrinsic_dynamic_pca,
        #     output_video_path=args.output_video_path,
        #     fps=10,
        # )
        
        # ==========================================================================================
        #  Distill moved features back to canonical grid & Visualize
        # ==========================================================================================
        if len(transforms) > 0:
             R_final, t_final = transforms[-1]
             
             # Distill
             distill_object_features(
                 grid=grid,
                 object_grid=object_grid,
                 decoder=decoder,
                 points_Y_local=points_Y,
                 R_final=R_final,
                 t_final=t_final,
                 device=device,
             )
             
             # Final Visualization with updated grid
             print("[MAIN] Visualizing updated canonical grid with moved particles...")
             
             # Construct final point cloud positions
             # Background (P \ Y)
             mask_Y = np.zeros(point_cloud.shape[0], dtype=bool)
             mask_Y[Y_indices] = True
             points_bg = point_cloud[~mask_Y]
             if points_bg.shape[1] > 3: 
                 points_bg = points_bg[:, :3]
             
             # Moved Object (Y transformed)
             points_Y_final = points_Y @ R_final.T + t_final
             
             points_final = np.vstack([points_bg, points_Y_final])
             
             visualize_pca_open3d(
                points_3d=points_final,
                grid=grid,
                decoder=decoder,
                device=device,
                camera_extrinsic=camera_extrinsic_dynamic_pca,
             )
             
    else:
        print("[MAIN] Skipping dynamic PCA because Y is empty.")


if __name__ == "__main__":
    main()
