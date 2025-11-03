import omnigibson as og
from omnigibson.macros import gm
import numpy as np
import imageio
import os
import argparse
import math
from omnigibson.utils.asset_utils import get_available_behavior_1k_scenes
from omnigibson.utils.ui_utils import choose_from_options
import omnigibson.utils.transform_utils as T
import helper  # helper.py 임포트
import open3d as o3d
import time
import copy
import viser
from scipy.spatial.transform import Rotation
import yaml
from tqdm import tqdm

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False


def save_data(frame_idx, rgb_image, depth_image, seg_instance_id_image, pos, orn, rgb_dir, depth_dir, seg_instance_id_dir, poses_dir):
    frame_filename_base = f"{frame_idx:05d}"
    imageio.imwrite(os.path.join(rgb_dir, f"{frame_filename_base}.png"), rgb_image)
    np.save(os.path.join(depth_dir, f"{frame_filename_base}.npy"), depth_image)
    np.save(os.path.join(seg_instance_id_dir, f"{frame_filename_base}.npy"), seg_instance_id_image)
    
    # Construct and save the extrinsic matrix in OpenCV format (world-to-camera)
    R_world_og = Rotation.from_quat(orn.cpu().numpy()).as_matrix()
    t_world_og = pos.cpu().numpy()
    
    R_og_world = R_world_og.T
    t_og_world = -R_og_world @ t_world_og
    
    T_og_world = np.eye(4)
    T_og_world[:3, :3] = R_og_world
    T_og_world[:3, 3] = t_og_world
    
    T_cv_og = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    extrinsic_cv = T_cv_og @ T_og_world
    np.save(os.path.join(poses_dir, f"{frame_filename_base}.npy"), extrinsic_cv)


def sample_random_orientation():
    # sample random orientation
    orientation = T.random_quaternion()
    return orientation

def main():
    """
    Renders and saves RGB, Depth, and Instance images from a set of sampled poses.
    """
    parser = argparse.ArgumentParser(description="Render images from sampled poses.")
    parser.add_argument("--points_file", type=str, default="house_single_floor_points.ply",
                        help="Path to the .ply file containing points.")
    parser.add_argument("--num_samples", type=int, default=30000,
                        help="Number of points to sample and render.")
    parser.add_argument("--scene", type=str, default="house_single_floor", help="Scene model to load. If not specified, will prompt for selection.")
    parser.add_argument("--activity_name", type=str, default="sorting_household_items", help="BEHAVIOR activity to load. If specified, objects for the task will be loaded.")
    parser.add_argument("--activity_definition_id", type=int, default=0, help="Definition ID for the activity.")
    parser.add_argument("--activity_instance_id", type=int, default=0, help="Instance ID for the activity.")
    args = parser.parse_args()
    
    server = viser.ViserServer()

    # Choose scene
    available_scenes = get_available_behavior_1k_scenes()
    if args.scene:
        if args.scene not in available_scenes:
            print(f"Error: Scene '{args.scene}' not found. Available scenes: {list(available_scenes)}")
            return
        scene_model = args.scene
    else:
        scene_model = choose_from_options(options=available_scenes, name="scene")
        
    # Create directories to save data
    data_dir = os.path.join("mapping/dataset", args.activity_name)
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    seg_instance_id_dir = os.path.join(data_dir, "seg_instance_id")
    poses_dir = os.path.join(data_dir, "poses")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(seg_instance_id_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    
    # Configuration for the environment
    config = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
        },
        "env": {
            "external_sensors": [{
                "sensor_type": "VisionSensor",
                "name": "my_sensor",
                "modalities": ["rgb", "depth_linear", "seg_instance_id"],
                # "modalities": ["rgb","depth_linear"],
                "sensor_kwargs": {
                    "image_height": 512,
                    "image_width": 512,
                },
                "position": [0, 0, 1.5],
                "orientation": [0, 0, 0, 1], 
            }]
        },
        "robots": [
            {
                "type": "Fetch",
            }
        ],
    }
    
    # If an activity is specified, load task-relevant objects
    if args.activity_name:
        config["scene"]["load_task_relevant_only"] = False
        # config["scene"]["not_load_object_categories"] = ["ceilings"]
        config["task"] = {
            "type": "BehaviorTask",
            "activity_name": args.activity_name,
            "activity_definition_id": args.activity_definition_id,
            "activity_instance_id": args.activity_instance_id,
            "online_object_sampling": False,
        }
    
    # Load the environment
    env = og.Environment(configs=config)
    robot = env.robots[0]
    robot.set_position_orientation(position=np.array([-1.0, -10.0, 2.0]))
    sensor = env._external_sensors["my_sensor"]
    
    # Load points
    points = o3d.io.read_point_cloud(args.points_file)
    points = np.asarray(points.points)

    # Sample points if num_samples is specified
    if 0 < args.num_samples < len(points):
        indices = np.random.choice(len(points), args.num_samples, replace=False)
        points = points[indices]
    
    for _ in range(5):
        og.sim.step()
    
    # Camera parameters from config and defaults
    image_width = config["env"]["external_sensors"][0]["sensor_kwargs"]["image_width"]
    image_height = config["env"]["external_sensors"][0]["sensor_kwargs"]["image_height"]
    focal_length = 17.0  # Default value from VisionSensor
    horizontal_aperture = 40.0  # Default value from VisionSensor

    # Calculate visualization parameters
    aspect_ratio = image_width / image_height
    vertical_aperture = horizontal_aperture * (image_height / image_width)
    vertical_fov = 2 * math.atan(vertical_aperture / (2 * focal_length))

    # Calculate and save intrinsic matrix
    cx = image_width / 2.0
    cy = image_height / 2.0
    fx = focal_length * image_width / horizontal_aperture
    fy = focal_length * image_height / vertical_aperture
    
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    intrinsics_path = os.path.join(data_dir, "intrinsics.txt")
    np.savetxt(intrinsics_path, intrinsic_matrix)

    all_world_points = []
    all_colors = []
    valid_sample_count = 0
    for i, point in enumerate(tqdm(points, desc="Processing points")):
        # 1. Get first sample for evaluation
        orientation = sample_random_orientation()
        
        sensor.set_position_orientation(position=point, orientation=orientation)
        for _ in range(5):
            og.sim.step()
        
        cur_pos, cur_orn = sensor.get_position_orientation()
        
        obs, _ = sensor.get_obs()
        rgb_image = copy.deepcopy(obs["rgb"][..., :3].cpu().numpy())
        depth_image = copy.deepcopy(obs["depth_linear"].cpu().numpy().squeeze())
        seg_instance_id_image = copy.deepcopy(obs["seg_instance_id"].cpu().numpy().squeeze())

        # Filter based on depth image - if view is mostly bad, skip
        total_pixels = depth_image.size
        too_close_pixels = np.sum(depth_image <= 0.15)
        too_far_pixels = np.sum(depth_image >= 10)
        if (too_far_pixels / total_pixels) > 0.5 or (too_close_pixels / total_pixels) > 0.5:
            print(f"  -> Skipping point {i} due to invalid depth values in initial sample.")
            continue
        
        # Save the initial sample if it passes the depth filter
        save_data(valid_sample_count, rgb_image, depth_image, seg_instance_id_image, cur_pos, cur_orn, rgb_dir, depth_dir, seg_instance_id_dir, poses_dir)
        valid_sample_count += 1
        time.sleep(0.01)
            
        # 2. Generate world point cloud to check condition
        points_camera, _ = helper.depth_to_point_cloud(depth_image, rgb_image, intrinsic_matrix)
        
        # Coordinate system transformation from OpenCV to OmniGibson sensor frame
        points_camera[:, 1] *= -1.0
        points_camera[:, 2] *= -1.0
        
        # world_points = helper.transform_point_cloud(points_camera, cur_pos, cur_orn).cpu().numpy()

        # 3. Check z-value condition
        # z_values = world_points[:, 2]
        # if len(z_values) == 0:
        #     print(f"  -> No valid points in point cloud for sample {i}. Skipping.")
        #     continue
            
        # valid_z_mask = (z_values >= 0.1) & (z_values <= 5.0)
        # valid_z_ratio = np.sum(valid_z_mask) / len(z_values)

        # if valid_z_ratio > 0.95:
        #     print(f"Found a good location at point {i}. Sampling 10 poses.")
        #     for j in range(10):
        #         new_orientation = sample_random_orientation()
        #         sensor.set_position_orientation(position=point, orientation=new_orientation)
        #         for _ in range(5):
        #             og.sim.step()

        #         new_pos, new_orn = sensor.get_position_orientation()
        #         obs, _ = sensor.get_obs()
        #         new_rgb_image = copy.deepcopy(obs["rgb"][..., :3].cpu().numpy())
        #         new_depth_image = copy.deepcopy(obs["depth_linear"].cpu().numpy().squeeze())
        #         new_seg_instance_id_image = copy.deepcopy(obs["seg_instance_id"].cpu().numpy().squeeze())

        #         # Filter this new sample based on depth image
        #         total_pixels = new_depth_image.size
        #         too_close_pixels = np.sum(new_depth_image <= 0.1)
        #         too_far_pixels = np.sum(new_depth_image >= 10)
        #         if (too_far_pixels / total_pixels) > 0.5 or (too_close_pixels / total_pixels) > 0.5:
        #             print(f"  -> Skipping sample {j} for point {i} due to invalid depth values.")
        #             continue
                
        #         # Save data
        #         save_data(valid_sample_count, new_rgb_image, new_depth_image, new_seg_instance_id_image, new_pos, new_orn, rgb_dir, depth_dir, seg_instance_id_dir, poses_dir)
                
        #         valid_sample_count += 1
        #         time.sleep(0.01)
        # else:
        #     print(f"  -> Point {i} is not good (z-ratio: {valid_z_ratio:.2f}). Skipping.")

    # Calculate and print point cloud bounds and visualize the full point cloud
    # if all_world_points:
    #     full_point_cloud = np.concatenate(all_world_points, axis=0)
    #     full_colors = np.concatenate(all_colors, axis=0)
        
    #     min_bounds = np.min(full_point_cloud, axis=0)
    #     max_bounds = np.max(full_point_cloud, axis=0)
    #     print("\n--- Point Cloud Bounds ---")
    #     print(f"X: {min_bounds[0]:.4f} to {max_bounds[0]:.4f}")
    #     print(f"Y: {min_bounds[1]:.4f} to {max_bounds[1]:.4f}")
    #     print(f"Z: {min_bounds[2]:.4f} to {max_bounds[2]:.4f}")
    #     print("--------------------------\n")

    #     # Downsample the full point cloud for visualization performance
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(full_point_cloud)
    #     pcd.colors = o3d.utility.Vector3dVector(full_colors)
    #     downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.1)

    #     # Add the full point cloud to viser
    #     server.add_point_cloud(
    #         name="/point_clouds/full",
    #         points=np.asarray(downsampled_pcd.points),
    #         colors=(np.asarray(downsampled_pcd.colors) * 255).astype(np.uint8),
    #     )

    # print("Viser visualization running. Open the URL in your browser.")
    # while True:
    #     time.sleep(1)


if __name__ == "__main__":
    main()