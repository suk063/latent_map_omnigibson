import open3d as o3d
import numpy as np
import os
import imageio.v2 as imageio
import argparse
import viser
import time

def main():
    """
    Visualizes an aggregated point cloud from an RGBD dataset.
    """
    parser = argparse.ArgumentParser(description="Visualize a point cloud from an RGBD dataset.")
    parser.add_argument("--data_dir", type=str, default="mapping/dataset", help="Path to the dataset directory.")
    args = parser.parse_args()

    data_dir = args.data_dir
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    poses_dir = os.path.join(data_dir, "poses")
    intrinsics_path = os.path.join(data_dir, "intrinsics.txt")

    # Load camera intrinsics
    try:
        intrinsic_matrix = np.loadtxt(intrinsics_path)
    except FileNotFoundError:
        print(f"Error: Intrinsics file not found at {intrinsics_path}")
        return
        
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    # Assuming image dimensions are 720x720 as in render_data.py
    intrinsics.set_intrinsics(
        width=720,
        height=720,
        fx=intrinsic_matrix[0, 0],
        fy=intrinsic_matrix[1, 1],
        cx=intrinsic_matrix[0, 2],
        cy=intrinsic_matrix[1, 2]
    )

    # Get file lists
    try:
        rgb_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.png')])
        depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.npy')])
        pose_files = sorted([os.path.join(poses_dir, f) for f in os.listdir(poses_dir) if f.endswith('.npy')])
    except FileNotFoundError as e:
        print(f"Error: Data directory not found or incomplete. {e}")
        return

    if not (len(rgb_files) == len(depth_files) == len(pose_files)):
        print("Error: The number of rgb, depth, and pose files must be the same.")
        return
    
    if not rgb_files:
        print("Error: No data found in the specified directory.")
        return

    all_pcds = []
    
    for i in range(0, len(rgb_files), 100):
        print(f"Processing frame {i+1}/{len(rgb_files)}: {os.path.basename(rgb_files[i])}")
        
        # Load data for the current frame
        rgb_image = imageio.imread(rgb_files[i])
        depth_image = np.load(depth_files[i])
        pose_world_to_cam = np.load(pose_files[i])

        # Create Open3D image objects
        color_o3d = o3d.geometry.Image(rgb_image)
        depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

        # Create an RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=100.0,  # Truncate depth values far away
            convert_rgb_to_intensity=False
        )

        # Create a point cloud from the RGBD image and intrinsics
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsics
        )
        
        # The pose is world-to-camera, so we need the inverse for camera-to-world
        pose_cam_to_world = np.linalg.inv(pose_world_to_cam)
        pcd.transform(pose_cam_to_world)
        
        all_pcds.append(pcd)

    print("Combining all point clouds...")
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in all_pcds:
        combined_pcd += pcd

    print("Downsampling the combined point cloud with a voxel size of 0.1...")
    downsampled_pcd = combined_pcd.voxel_down_sample(voxel_size=0.1)

    print("Visualizing the final point cloud with Viser...")
    server = viser.ViserServer()
    points = np.asarray(downsampled_pcd.points)
    colors = (np.asarray(downsampled_pcd.colors) * 255).astype(np.uint8)
    
    server.add_point_cloud(
        name="/point_cloud",
        points=points,
        colors=colors,
    )
    
    print("Viewer running. Open http://localhost:8080 in your browser.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
