import numpy as np
import os
import imageio.v2 as imageio
import argparse
import cv2
import torch
import sys

import plotly.graph_objects as go
import plotly.io as pio

# local co-tracker repo (for imports etc. if needed)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "co-tracker"))


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

def select_keypoints(rgb_image: np.ndarray):
    """
    Allows the user to select multiple keypoints with the mouse on the first frame.
    - Left click: add a keypoint
    - ENTER: finish selection
    - ESC: cancel (returns empty list)

    Returns:
        List[(u, v)]
    """
    state = {"pts": []}

    img_disp = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append((x, y))
            cv2.circle(img_disp, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select Keypoints", img_disp)

    cv2.namedWindow("Select Keypoints", cv2.WINDOW_NORMAL)
    cv2.imshow("Select Keypoints", img_disp)
    cv2.setMouseCallback("Select Keypoints", mouse_callback)

    print(
        "[INFO] Left-click to add keypoints. Press ENTER when done, or ESC to cancel."
    )
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            state["pts"].clear()
            break
        # ENTER key (13 on many systems, sometimes 10)
        if key in (13, 10):
            break

    cv2.destroyWindow("Select Keypoints")
    return state["pts"]


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
                chunk = video[:, ind : ind + 2 * step]  # length <= 2*step
                pred_tracks, pred_visibility = cotracker(
                    video_chunk=chunk,
                    grid_size=grid_size,
                    queries=queries,
                    add_support_grid=True,
                )
                # 온라인 모드는 내부 state를 유지하므로,
                # 마지막 호출의 pred_tracks에 전체 트랙이 들어있다고 가정.

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
    vis_threshold: float = 0.5,
    point_stride: int = 4,       # stride for dense point cloud
):
    """
    Compute 3D positions and RGB colors for each keypoint at each time step,
    AND dense RGB point cloud for entire image at each time step.

    Returns:
        positions_3d_kp: (T_eff, N, 3) with NaN where invalid (keypoints)
        colors_rgb_kp:   (T_eff, N, 3) uint8, with 0 where invalid
        pc_positions:    list of length T_eff, each (M_t, 3) dense point cloud
        pc_colors:       list of length T_eff, each (M_t, 3) uint8 RGB
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

        # -----------------------------
        # Dense RGB point cloud (whole image)
        # -----------------------------
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

        # -----------------------------
        # Keypoint 3D trajectories
        # -----------------------------
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

            # Store 3D position
            positions_3d[t, n, :] = p_world

            # Get RGB color at corresponding pixel
            u_i = int(round(u))
            v_i = int(round(v))
            if 0 <= u_i < W and 0 <= v_i < H:
                colors_rgb[t, n, :] = rgb_frame[v_i, u_i, :]  # RGB

    return positions_3d, colors_rgb, pc_positions, pc_colors


# ==============================================================================================
#  Plotly 3D animation:
#   - per-step dense RGB point cloud (whole image)
#   - cumulative keypoint trajectories 0..t
#   - trajectory color depends on time step t
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
    kp_colors_rgb: np.ndarray,     # (T, N, 3) uint8  (현재는 사용 X, 있어도 됨)
    pc_positions: list[np.ndarray],  # len T, each (M_t, 3)
    pc_colors: list[np.ndarray],     # len T, each (M_t, 3)
    output_html: str,
):
    """
    For each time step t, visualize:
      - dense RGB point cloud at time t (whole image)
      - keypoint trajectories 0..t for each keypoint (cumulative)
        BUT: within the trajectory, each point is colored by its own time step.
    """
    T, N, _ = kp_positions_3d.shape

    # --- Initial data (t = 0) ---
    t0 = 0
    pts0_pc = pc_positions[t0]   # (M0, 3)
    cols0_pc = pc_colors[t0]     # (M0, 3)
    pc_color_strings0 = [
        f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in cols0_pc
    ]

    data = []

    # 0) dense point cloud at t0
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

    # 1..N) trajectories for each keypoint at t0
    # t=0 이라서 사실 점 1개뿐이지만, 동일한 로직 유지
    for n in range(N):
        pts = kp_positions_3d[: t0 + 1, n, :]  # (1, 3) at t0
        mask = ~np.isnan(pts[:, 0])
        pts_valid = pts[mask]
        if pts_valid.shape[0] > 0:
            x_n = pts_valid[:, 0]
            y_n = pts_valid[:, 1]
            z_n = pts_valid[:, 2]
            times = np.arange(0, t0 + 1)[mask]  # -> [0]
        else:
            x_n = y_n = z_n = []
            times = []

        data.append(
            go.Scatter3d(
                x=x_n,
                y=y_n,
                z=z_n,
                mode="lines+markers",
                # line은 얇고 연한 고정색
                line=dict(width=3, color="rgba(220,220,220,0.8)"),
                # marker는 time step에 따라 colorscale로 색을 다르게
                marker=dict(
                    size=4,
                    color=times,          # 각 point의 시간 index
                    colorscale="Viridis", # 아무거나 좋은 colormap
                    cmin=0,
                    cmax=T - 1,
                    showscale=True if n == 0 and t0 == 0 else False,  # 컬러바는 딱 한 번만
                ),
                name=f"keypoint_{n}",
                showlegend=True if n == 0 else False,  # legend는 하나만
            )
        )

    # --- Frames ---
    frames = []
    for t in range(T):
        pts_t_pc = pc_positions[t]
        cols_t_pc = pc_colors[t]
        pc_color_strings_t = [
            f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in cols_t_pc
        ]

        frame_data = []

        # dense point cloud at time t
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

        # cumulative trajectories per keypoint (0..t)
        for n in range(N):
            pts = kp_positions_3d[: t + 1, n, :]  # (t+1, 3)
            mask = ~np.isnan(pts[:, 0])
            pts_valid = pts[mask]
            if pts_valid.shape[0] > 0:
                x_n = pts_valid[:, 0]
                y_n = pts_valid[:, 1]
                z_n = pts_valid[:, 2]
                times = np.arange(0, t + 1)[mask]  # 각 point의 time index
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
                        showscale=False,  # 컬러바는 초기 data에만
                    ),
                    name=f"keypoint_{n}",
                    showlegend=False,
                )
            )

        frames.append(go.Frame(data=frame_data, name=str(t)))

    # --- Layout with slider & play button ---
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
#  Main
# ==============================================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Track multiple keypoints in 3D using RGB, Depth, Pose and "
            "CoTracker3 (online), and visualize per-step dense RGB point cloud + "
            "cumulative trajectories with Plotly 3D (HTML animation)."
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
        default=970,
        help="The starting frame index (0-based).",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=2870,
        help="The ending frame index (exclusive). -1 means till the last frame.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for CoTracker3: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default="trajectories_animation.html",
        help="Output HTML file path for Plotly 3D visualization.",
    )
    parser.add_argument(
        "--pc_stride",
        type=int,
        default=4,
        help="Pixel stride for dense point cloud (larger = fewer points).",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=5,
        help="Process every N-th frame for tracking (e.g., 10 for every 10th frame).",
    )
    args = parser.parse_args()

    # Device check
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    # --------------------------------------------------------------------------- #
    #  Set up paths
    # --------------------------------------------------------------------------- #
    data_dir = os.path.join(args.data_dir, args.camera_name)
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    poses_dir = os.path.join(data_dir, "poses")
    intrinsics_path = os.path.join(data_dir, "intrinsics.txt")

    # --------------------------------------------------------------------------- #
    #  Load intrinsics
    # --------------------------------------------------------------------------- #
    if not os.path.exists(intrinsics_path):
        print(f"[ERROR] Intrinsics file not found at {intrinsics_path}")
        return

    intrinsic_matrix = np.loadtxt(intrinsics_path)
    fx_orig = intrinsic_matrix[0, 0]
    fy_orig = intrinsic_matrix[1, 1]
    cx_orig = intrinsic_matrix[0, 2]
    cy_orig = intrinsic_matrix[1, 2]
    print(f"[INIT] Loaded intrinsics from {intrinsics_path}")

    # --------------------------------------------------------------------------- #
    #  Load file lists (RGB, depth, pose)
    # --------------------------------------------------------------------------- #
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

    # Set end_frame
    if args.end_frame == -1 or args.end_frame > num_frames:
        end_frame = num_frames
    else:
        end_frame = args.end_frame

    start = args.start_frame
    if start < 0 or start >= end_frame:
        print(
            f"[ERROR] Invalid start_frame {start}. It must be in [0, {end_frame-1}]."
        )
        return

    print(f"[INFO] Using frames from {start} to {end_frame - 1} (inclusive).")

    # Frame sampling
    frame_indices = list(range(start, end_frame, args.frame_step))
    if not frame_indices:
        print("[ERROR] No frames selected with the given start, end, and step. Exiting.")
        return

    # Subsample file lists based on frame_indices relative to the original full lists
    subsampled_rgb_files = [rgb_files[i] for i in frame_indices]
    subsampled_depth_files = [depth_files[i] for i in frame_indices]
    subsampled_pose_files = [pose_files[i] for i in frame_indices]

    print(f"[INFO] Sub-sampling every {args.frame_step} frames. Total frames to process: {len(frame_indices)}")

    # --------------------------------------------------------------------------- #
    #  Load first frame & select multiple keypoints
    # --------------------------------------------------------------------------- #
    first_rgb = imageio.imread(rgb_files[start]) # Keypoints are selected on the original start frame

    h_rgb, w_rgb, _ = first_rgb.shape
    first_depth = np.load(depth_files[start])
    h_d, w_d = first_depth.shape

    if (h_rgb, w_rgb) != (h_d, w_d):
        print(
            f"[WARN] RGB size {first_rgb.shape} and depth size {first_depth.shape} differ. "
            "Assuming intrinsics correspond to depth resolution. "
            "If not, you may need resizing / scaling."
        )

    print(
        f"[INFO] Selecting keypoints on frame {start}: {os.path.basename(rgb_files[start])}"
    )
    keypoints = select_keypoints(first_rgb)
    if len(keypoints) == 0:
        print("[ERROR] No keypoints selected. Exiting.")
        return

    print(f"[INFO] Selected {len(keypoints)} keypoints: {keypoints}")

    # --------------------------------------------------------------------------- #
    #  Build video tensor for CoTracker3 (frames start .. end_frame-1)
    # --------------------------------------------------------------------------- #
    frames_list = []
    for rgb_file_path in subsampled_rgb_files:
        rgb = imageio.imread(rgb_file_path)
        frames_list.append(rgb)

    frames_np = np.stack(frames_list, axis=0)  # (T, H, W, 3)
    T = frames_np.shape[0]
    print(f"[INIT] Loaded {T} RGB frames for CoTracker3.")

    # --------------------------------------------------------------------------- #
    #  Run CoTracker3 (online) to get 2D tracks for all keypoints
    # --------------------------------------------------------------------------- #
    print("[TRACK] Running CoTracker3 online tracking (multi-keypoint)...")
    tracks_xy, visibilities = run_cotracker_online_multi(
        frames_np, keypoints, device=device
    )
    if tracks_xy.shape[0] != T:
        print(
            f"[WARN] CoTracker returned {tracks_xy.shape[0]} frames but input has {T}. "
            "Will use the min length."
        )

    # --------------------------------------------------------------------------- #
    #  Compute 3D positions + RGB colors over time (keypoints) + dense point clouds
    # --------------------------------------------------------------------------- #
    positions_3d, colors_rgb, pc_positions, pc_colors = compute_3d_positions_and_colors_over_time(
        frames_np=frames_np,
        tracks_xy=tracks_xy,
        visibilities=visibilities,
        depth_files=subsampled_depth_files,
        pose_files=subsampled_pose_files,
        start_frame_index=0,  # The lists are already subsampled
        fx=fx_orig,
        fy=fy_orig,
        cx=cx_orig,
        cy=cy_orig,
        vis_threshold=0.5,
        point_stride=args.pc_stride,
    )

    # --------------------------------------------------------------------------- #
    #  Plotly 3D animation: dense point cloud + cumulative trajectories
    # --------------------------------------------------------------------------- #
    visualize_animation_plotly(
        kp_positions_3d=positions_3d,
        kp_colors_rgb=colors_rgb,
        pc_positions=pc_positions,
        pc_colors=pc_colors,
        output_html=args.output_html,
    )


if __name__ == "__main__":
    main()
