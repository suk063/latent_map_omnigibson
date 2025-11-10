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
    parser.add_argument("--data_dir", type=str, default="DATASETS/behavior/processed_data/task-0021/episode_00210170/", help="Path to the dataset directory.")
    args = parser.parse_args()

    data_dir = args.data_dir
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    poses_dir = os.path.join(data_dir, "poses")
    seg_instance_id_dir = os.path.join(data_dir, "seg_instance_id")
    intrinsics_path = os.path.join(data_dir, "intrinsics.txt")

    # Load camera intrinsics
    try:
        intrinsic_matrix = np.loadtxt(intrinsics_path)
    except FileNotFoundError:
        print(f"Error: Intrinsics file not found at {intrinsics_path}")
        return
        
    fx=intrinsic_matrix[0, 0]
    fy=intrinsic_matrix[1, 1]
    cx=intrinsic_matrix[0, 2]
    cy=intrinsic_matrix[1, 2]

    # Get file lists
    try:
        rgb_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.png')])
        depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.npy')])
        pose_files = sorted([os.path.join(poses_dir, f) for f in os.listdir(poses_dir) if f.endswith('.npy')])
        seg_instance_id_files = sorted([os.path.join(seg_instance_id_dir, f) for f in os.listdir(seg_instance_id_dir) if f.endswith('.npy')])
    except FileNotFoundError as e:
        print(f"Error: Data directory not found or incomplete. {e}")
        return

    if not (len(rgb_files) == len(depth_files) == len(pose_files) == len(seg_instance_id_files)):
        print("Error: The number of rgb, depth, pose, and instance segmentation files must be the same.")
        return
    
    if not rgb_files:
        print("Error: No data found in the specified directory.")
        return

    all_pcds_rgb = []
    all_points_instance_list = []
    all_colors_instance_list = []
    all_ids_instance_list = []
    instance_color_map = {}
    
    for i in range(0, len(rgb_files), 100):
        print(f"Processing frame {i+1}/{len(rgb_files)}: {os.path.basename(rgb_files[i])}")
        
        # Load data for the current frame
        rgb_image = imageio.imread(rgb_files[i])
        depth_image = np.load(depth_files[i])
        pose_world_to_cam = np.load(pose_files[i])
        seg_instance_id_image = np.load(seg_instance_id_files[i])

        h, w, _ = rgb_image.shape
        v, u = np.indices((h, w))
        
        mask = (depth_image > 0) & (depth_image < 100.0)
        
        depth_valid = depth_image[mask]
        u_valid = u[mask]
        v_valid = v[mask]
        rgb_valid = rgb_image[mask] / 255.0
        seg_ids_valid = seg_instance_id_image[mask]
        
        z = depth_valid
        x = (u_valid - cx) * z / fx
        y = (v_valid - cy) * z / fy
        points = np.stack((x, y, z), axis=-1)
        
        # RGB Point Cloud
        pcd_rgb = o3d.geometry.PointCloud()
        pcd_rgb.points = o3d.utility.Vector3dVector(points)
        pcd_rgb.colors = o3d.utility.Vector3dVector(rgb_valid)
        
        # Instance Point Cloud
        unique_ids, inverse_indices = np.unique(seg_ids_valid, return_inverse=True)
        unique_colors = np.zeros((len(unique_ids), 3))
        for idx, uid in enumerate(unique_ids):
            if uid not in instance_color_map:
                if uid == 0:
                    instance_color_map[uid] = np.array([0.5, 0.5, 0.5])
                else:
                    instance_color_map[uid] = np.random.rand(3)
            unique_colors[idx] = instance_color_map[uid]
        instance_colors = unique_colors[inverse_indices]
        
        pcd_instance = o3d.geometry.PointCloud()
        pcd_instance.points = o3d.utility.Vector3dVector(points)
        pcd_instance.colors = o3d.utility.Vector3dVector(instance_colors)
        
        # The pose is world-to-camera, so we need the inverse for camera-to-world
        pose_cam_to_world = np.linalg.inv(pose_world_to_cam)
        pcd_rgb.transform(pose_cam_to_world)
        pcd_instance.transform(pose_cam_to_world)
        
        all_pcds_rgb.append(pcd_rgb)
        all_points_instance_list.append(np.asarray(pcd_instance.points))
        all_colors_instance_list.append(instance_colors)
        all_ids_instance_list.append(seg_ids_valid)

    print("Combining all RGB point clouds...")
    combined_pcd_rgb = o3d.geometry.PointCloud()
    for pcd in all_pcds_rgb:
        combined_pcd_rgb += pcd

    print("Downsampling the combined RGB point cloud with a voxel size of 0.03...")
    downsampled_pcd_rgb = combined_pcd_rgb.voxel_down_sample(voxel_size=0.03)

    # Calculate and print the bounding box
    bbox = downsampled_pcd_rgb.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    print(f"Point cloud bounding box:\n  Min bound: {min_bound}\n  Max bound: {max_bound}")

    print("Generating Plotly 3D visualization for RGB and saving to HTML...")
    points_rgb = np.asarray(downsampled_pcd_rgb.points)
    colors_rgb = np.asarray(downsampled_pcd_rgb.colors)
    
    fig_rgb = go.Figure(data=[go.Scatter3d(
        x=points_rgb[:, 0],
        y=points_rgb[:, 1],
        z=points_rgb[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=colors_rgb,
            opacity=0.8
        )
    )])

    fig_rgb.update_layout(
        title='Point Cloud (RGB)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    output_filename_rgb = "point_cloud.html"
    fig_rgb.write_html(output_filename_rgb)
    print(f"RGB point cloud visualization saved to {os.path.abspath(output_filename_rgb)}")

    # Save the (x, y) coordinates to a numpy file
    xy_coords = points_rgb[:, :2]
    xy_filename = "point_cloud_xy.npy"
    np.save(xy_filename, xy_coords)
    print(f"(x, y) coordinates saved to {os.path.abspath(xy_filename)}")

    # Process and visualize instance point cloud
    print("\nCombining all instance point clouds...")
    combined_points_instance = np.concatenate(all_points_instance_list, axis=0)
    combined_colors_instance = np.concatenate(all_colors_instance_list, axis=0)
    combined_ids_instance = np.concatenate(all_ids_instance_list, axis=0)

    print("Downsampling the combined instance point cloud with a voxel size of 0.03...")
    voxel_size = 0.03
    voxel_indices = np.floor(combined_points_instance / voxel_size).astype(int)
    _, first_indices = np.unique(voxel_indices, axis=0, return_index=True)
    
    points_instance = combined_points_instance[first_indices]
    colors_instance = combined_colors_instance[first_indices]
    ids_instance = combined_ids_instance[first_indices]

    print("Generating Plotly 3D visualization for instance segmentation and saving to HTML...")
    
    fig_instance = go.Figure(data=[go.Scatter3d(
        x=points_instance[:, 0],
        y=points_instance[:, 1],
        z=points_instance[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=colors_instance,
            opacity=0.8
        ),
        customdata=ids_instance,
        hovertemplate='<b>Instance ID:</b> %{customdata}<br>' +
                      '<b>x:</b> %{x:.2f}<br>' +
                      '<b>y:</b> %{y:.2f}<br>' +
                      '<b>z:</b> %{z:.2f}<extra></extra>'
    )])

    fig_instance.update_layout(
        title='Point Cloud (Instance Segmentation)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    output_filename_instance = "point_cloud_instance.html"
    fig_instance.write_html(output_filename_instance)
    print(f"Instance point cloud visualization saved to {os.path.abspath(output_filename_instance)}")


if __name__ == "__main__":
    main()
