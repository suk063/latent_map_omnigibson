import open3d as o3d
import numpy as np
import os
import imageio.v2 as imageio
import argparse
import plotly.graph_objects as go

def main():
    """
    Visualizes an aggregated point cloud from an RGBD dataset.
    """
    parser = argparse.ArgumentParser(description="Visualize a point cloud from an RGBD dataset.")
    parser.add_argument("--data_dir", type=str, default="DATASETS/behavior/processed_data/task-0027/episode_00270010/", help="Path to the dataset directory.")
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
        width=512,
        height=512,
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

    # if not (len(rgb_files) == len(depth_files) == len(pose_files)):
    #     print("Error: The number of rgb, depth, and pose files must be the same.")
    #     return
    
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

    print("Downsampling the combined point cloud with a voxel size of 0.02...")
    downsampled_pcd = combined_pcd.voxel_down_sample(voxel_size=0.03)

    # Calculate and print the bounding box
    bbox = downsampled_pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    print(f"Point cloud bounding box:\n  Min bound: {min_bound}\n  Max bound: {max_bound}")

    print("Generating Plotly 3D visualization and saving to HTML...")
    points = np.asarray(downsampled_pcd.points)
    colors = np.asarray(downsampled_pcd.colors)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=colors,
            opacity=0.8
        )
    )])

    fig.update_layout(
        title='Point Cloud',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    output_filename = "point_cloud.html"
    fig.write_html(output_filename)
    print(f"Point cloud visualization saved to {os.path.abspath(output_filename)}")

    # Save the (x, y) coordinates to a numpy file
    xy_coords = points[:, :2]
    xy_filename = "point_cloud_xy.npy"
    np.save(xy_filename, xy_coords)
    print(f"(x, y) coordinates saved to {os.path.abspath(xy_filename)}")


if __name__ == "__main__":
    main()
