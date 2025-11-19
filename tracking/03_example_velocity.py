import open3d as o3d
import numpy as np
import os
import imageio.v2 as imageio
import argparse
import cv2
from pathlib import Path
import viser


def compute_3d_point_from_depth(
    u: float,
    v: float,
    depth_map: np.ndarray,
    pose_world_to_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
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
    p_cam = np.array([x_cam, y_cam, z])

    # cam -> world transformation
    pose_cam_to_world = np.linalg.inv(pose_world_to_cam)
    R_c2w = pose_cam_to_world[:3, :3]
    t_c2w = pose_cam_to_world[:3, 3]
    p_world = R_c2w @ p_cam + t_c2w
    return p_world


def select_keypoint(rgb_image: np.ndarray):
    """
    Allows the user to select a single keypoint with the mouse on the first frame.
    Returns the selected pixel (u, v).
    """
    selected = {"pt": None}

    img_disp = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected["pt"] = (x, y)
            cv2.circle(img_disp, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select Keypoint", img_disp)

    cv2.namedWindow("Select Keypoint", cv2.WINDOW_NORMAL)
    cv2.imshow("Select Keypoint", img_disp)
    cv2.setMouseCallback("Select Keypoint", mouse_callback)

    print("[INFO] Click a point on the image to select keypoint (ESC to cancel).")
    while selected["pt"] is None:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyWindow("Select Keypoint")
    return selected["pt"]


def main():
    parser = argparse.ArgumentParser(
        description="Track a single keypoint in 3D using RGB, Depth, Optical Flow, and Pose."
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
        default=1070,
        help="The ending frame index (exclusive). -1 means till the last frame.",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------------------- #
    #  Set up paths
    # --------------------------------------------------------------------------- #
    data_dir = os.path.join(args.data_dir, args.camera_name)
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    poses_dir = os.path.join(data_dir, "poses")
    flow_dir = os.path.join(data_dir, "flow")
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
    #  Load file lists
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
        flow_files = sorted(
            [
                os.path.join(flow_dir, f)
                for f in os.listdir(flow_dir)
                if f.endswith(".npy")
            ]
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Data directory not found or incomplete. {e}")
        return

    if not (rgb_files and depth_files and pose_files and flow_files):
        print("[ERROR] One of rgb/depth/pose/flow is missing or empty.")
        return

    # Set end_frame
    num_frames = len(rgb_files)
    if args.end_frame == -1 or args.end_frame > num_frames:
        end_frame = num_frames
    else:
        end_frame = args.end_frame

    if end_frame - 1 > len(flow_files):
        print(
            f"[WARN] Number of flow files ({len(flow_files)}) is less than frames-1 ({end_frame-1}). "
            "Trajectory will stop when flow runs out."
        )
        end_for_flow = len(flow_files)
    else:
        end_for_flow = end_frame - 1  # The last flow index is end_frame-1

    start = args.start_frame
    if start < 0 or start >= end_for_flow:
        print(
            f"[ERROR] Invalid start_frame {start}. It must be in [0, {end_for_flow-1}]."
        )
        return

    print(
        f"[INFO] Using frames from {start} to {end_for_flow} (flow from i -> i+1 for i in this range)."
    )

    # --------------------------------------------------------------------------- #
    #  Load first frame & select keypoint
    # --------------------------------------------------------------------------- #
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

    print(f"[INFO] Selecting keypoint on frame {start}: {os.path.basename(rgb_files[start])}")
    keypoint = select_keypoint(first_rgb)
    if keypoint is None:
        print("[ERROR] No keypoint selected. Exiting.")
        return

    u, v = keypoint
    print(f"[INFO] Selected keypoint at pixel (u={u}, v={v})")

    # --------------------------------------------------------------------------- #
    #  Compute 3D trajectory
    # --------------------------------------------------------------------------- #
    trajectory_world = []  # list of (3,) np.ndarray
    trajectory_pixels = []  # list of (u, v)

    for idx in range(start, end_for_flow + 1):  # Also compute 3D for the last frame
        rgb_path = rgb_files[idx]
        depth_path = depth_files[idx]
        pose_path = pose_files[idx]

        depth = np.load(depth_path)
        pose_world_to_cam = np.load(pose_path)

        # Compute 3D point in the current frame
        p_world = compute_3d_point_from_depth(
            u=u,
            v=v,
            depth_map=depth,
            pose_world_to_cam=pose_world_to_cam,
            fx=fx_orig,
            fy=fy_orig,
            cx=cx_orig,
            cy=cy_orig,
        )

        if p_world is None:
            print(
                f"[WARN] Invalid depth or out-of-bounds at frame {idx}, pixel ({u:.2f}, {v:.2f}). Stopping tracking."
            )
            break

        trajectory_world.append(p_world)
        trajectory_pixels.append((u, v))
        print(
            f"[TRACK] Frame {idx}: pixel=({u:.2f}, {v:.2f}), world={p_world.tolist()}"
        )

        # If it's the last frame, don't apply flow anymore
        if idx == end_for_flow:
            break

        # Update pixel location for the next frame (optical flow)
        flow = np.load(flow_files[idx])  # flow from idx -> idx+1
        if flow.shape[:2] != depth.shape:
            print(
                f"[WARN] Flow shape {flow.shape} and depth shape {depth.shape} differ at frame {idx}. "
                "Stopping tracking."
            )
            break

        v_i = int(round(v))
        u_i = int(round(u))
        if (
            u_i < 0
            or u_i >= flow.shape[1]
            or v_i < 0
            or v_i >= flow.shape[0]
        ):
            print(
                f"[WARN] Pixel ({u:.2f}, {v:.2f}) out of flow bounds at frame {idx}. Stopping tracking."
            )
            break

        du, dv = flow[v_i, u_i, 0], flow[v_i, u_i, 1]
        u = u + du
        v = v + dv

    if len(trajectory_world) == 0:
        print("[ERROR] No valid trajectory points computed.")
        return

    traj_np = np.stack(trajectory_world, axis=0)  # (N, 3)
    print(f"[INFO] Computed trajectory with {traj_np.shape[0]} points.")

    # --------------------------------------------------------------------------- #
    #  Visualize 3D trajectory with Open3D
    # --------------------------------------------------------------------------- #
    geometries = []
    if traj_np.shape[0] == 1:
        # If there's only one point, use PointCloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(traj_np)
        geometries.append(pc)
    else:
        # Connect with LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(traj_np)
        lines = [[i, i + 1] for i in range(traj_np.shape[0] - 1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = [[1.0, 0.0, 0.0] for _ in lines]  # red
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

        # If you want to see the point cloud as well, add it
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(traj_np)
        geometries.append(pc)

    print("[VIS] Showing trajectory in Open3D window...")
    o3d.visualization.draw_geometries(geometries)

    # --------------------------------------------------------------------------- #
    #  Visualize 3D trajectory with viser
    # --------------------------------------------------------------------------- #
    print("[VIS] Starting viser server for trajectory visualization...")
    server = viser.ViserServer()

    # Add as a point cloud
    num_points = traj_np.shape[0]
    if num_points > 1:
        colors = np.zeros((num_points, 3), dtype=np.uint8)
        for i in range(num_points):
            # Gradient from red (start) to blue (end)
            ratio = i / (num_points - 1)
            colors[i, 0] = int(255 * (1.0 - ratio))  # R
            colors[i, 2] = int(255 * ratio)  # B
    else:
        colors = np.array([[255, 0, 0]], dtype=np.uint8)  # Red for a single point

    server.add_point_cloud(
        name="trajectory_points",
        points=traj_np.astype(np.float32),
        colors=colors,
        point_size=0.02,
    )

    # Also add a simple polyline (if supported)
    try:
        server.add_polyline(
            name="trajectory_line",
            points=traj_np.astype(np.float32),
            color=(255, 255, 255),
        )
    except Exception as e:
        print(f"[WARN] Could not add polyline to viser (maybe older version): {e}")

    print(
        "[VIS] Trajectory visualized with viser. "
        "Open the URL printed by viser (e.g., http://localhost:8080) in your browser."
    )

    # Wait to prevent the server from shutting down immediately
    try:
        import time

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[INFO] Exiting.")


if __name__ == "__main__":
    main()
