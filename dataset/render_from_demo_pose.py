import os
import torch as th
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.learning.utils.dataset_utils import makedirs_with_mode
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    TASK_INDICES_TO_NAMES,
)
from omnigibson.utils.ui_utils import create_module_logger
import numpy as np
import imageio
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import copy
import argparse
import time
import json
from omnigibson.macros import gm


log = create_module_logger(module_name="replay_obs")
log.setLevel(20)

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128
gm.ENABLE_TRANSITION_RULES = False

def get_pose_from_extrinsic(extrinsic_cv):
    """
    Converts an extrinsic matrix in OpenCV format back to position and orientation.
    """
    T_cv_og = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    T_og_world = T_cv_og @ extrinsic_cv
    
    R_og_world = T_og_world[:3, :3]
    t_og_world = T_og_world[:3, 3]
    
    R_world_og = R_og_world.T
    t_world_og = -R_world_og @ t_og_world
    
    pos = t_world_og

    # The pose files were saved with a scrambled quaternion.
    # An OmniGibson quat [w, x, y, z] was misinterpreted by SciPy as [x, y, z, w].
    # So, converting the saved matrix back to a SciPy quat ([x', y', z', w'])
    # actually gives us the original OG quat components: [w, x, y, z].
    scipy_quat_scrambled = Rotation.from_matrix(R_world_og).as_quat()

    # This is already in the [w, x, y, z] format that OmniGibson expects.
    correct_og_quat = np.array(scipy_quat_scrambled)
    
    return pos, correct_og_quat

def load_pose_files(data_folder, task_id, demo_id):
    """
    Loads all pose files for head, left_wrist, and right_wrist.
    """
    pose_files = []
    base_dir = os.path.join(data_folder, "processed_data", f"task-{task_id:04d}", f"episode_{demo_id:08d}")
    cam_keys = ["head", "left_wrist", "right_wrist"]
    for cam_key in cam_keys:
        poses_dir = os.path.join(base_dir, cam_key, "poses")
        if os.path.exists(poses_dir):
            files = [os.path.join(poses_dir, f) for f in os.listdir(poses_dir) if f.endswith('.npy')]
            pose_files.extend(files)
            print(f"Loaded {len(files)} poses from {cam_key}")
    return pose_files

def save_data(frame_idx, rgb_image, depth_image, seg_instance_id_image, pos, orn, rgb_dir, depth_dir, seg_instance_id_dir, poses_dir, seg_semantic_image=None, seg_semantic_dir=None, seg_instance_image=None, seg_instance_dir=None):
    frame_filename_base = f"{frame_idx:05d}"
    imageio.imwrite(os.path.join(rgb_dir, f"{frame_filename_base}.png"), rgb_image)
    np.save(os.path.join(depth_dir, f"{frame_filename_base}.npy"), depth_image)
    np.save(os.path.join(seg_instance_id_dir, f"{frame_filename_base}.npy"), seg_instance_id_image)
    if seg_semantic_image is not None and seg_semantic_dir is not None:
        np.save(os.path.join(seg_semantic_dir, f"{frame_filename_base}.npy"), seg_semantic_image)
    if seg_instance_image is not None and seg_instance_dir is not None:
        np.save(os.path.join(seg_instance_dir, f"{frame_filename_base}.npy"), seg_instance_image)
    
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

def sample_point(xy_points):
    """
    Samples a single (x, y, z) point.
    """
    idx = np.random.choice(xy_points.shape[0])
    xy = xy_points[idx]
    z = np.random.uniform(0.0, 2.0)
    point = np.array([xy[0], xy[1], z], dtype=np.float32)
    return point

def check_collision(point, radius):
    """
    Checks for collision at a given point with a given radius.
    """
    return og.sim.psqi.overlap_sphere_any(radius=radius, pos=point)

class BehaviorDataPlaybackWrapper(DataPlaybackWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # These will be set manually after creation
        self.rgb_dir = None
        self.depth_dir = None
        self.seg_instance_id_dir = None
        self.poses_dir = None
        self.head_cam_name = None
        self.frame_idx = -1

    def reset(self, *args, **kwargs):
        self.frame_idx = -1
        return super().reset(*args, **kwargs)

    def _process_obs(self, obs, info):
        # Increment frame counter first
        self.frame_idx += 1
    
        # --- Start of logic based on replay_obs.py ---
        robot = self.env.robots[0]
        base_pose = robot.get_position_orientation()
        cam_rel_poses = []
        head_cam_world_pose = None
        
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            # Standard processing from replay_obs.py
            assert camera_name.split("::")[1] in robot.sensors, f"Camera {camera_name} not found in robot sensors"
            if f"{camera_name}::seg_semantic" in obs:
                obs.pop(f"{camera_name}::seg_semantic")
            if f"{camera_name}::seg_instance_id" in obs:
                obs[f"{camera_name}::seg_instance_id"] = obs[f"{camera_name}::seg_instance_id"].cpu()

            # Get world pose for this camera
            cam_pose = robot.sensors[camera_name.split("::")[1]].get_position_orientation()
            
            # If this is the head camera, store its world pose for saving later.
            # We do NOT add this to the obs dict, to avoid breaking the hdf5 saving.
            if self.head_cam_name is not None and camera_name == self.head_cam_name:
                head_cam_world_pose = cam_pose

            # Calculate relative pose and add to list (this is saved in HDF5)
            cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
            
        # This key is expected by other parts of the system, so we add it.
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        # --- End of logic based on replay_obs.py ---
        
        # Now, save the frame data if the dirs are set
        if self.rgb_dir is not None:
            assert head_cam_world_pose is not None, "Head camera pose was not found during obs processing."
            log.info(f"Saving frame {self.frame_idx}...")
            # Replicate the logic from render_data.py: let imageio handle the datatype and color space.
            rgb_image = obs[f"{self.head_cam_name}::rgb"][..., :3].cpu().numpy().copy()
            depth = obs[f"{self.head_cam_name}::depth_linear"].cpu().numpy().squeeze()
            seg = obs[f"{self.head_cam_name}::seg_instance_id"].cpu().numpy().squeeze()
            pos, orn = head_cam_world_pose
            save_data(self.frame_idx, rgb_image, depth, seg, pos, orn, self.rgb_dir, self.depth_dir, self.seg_instance_id_dir, self.poses_dir)

        return obs

def build_scene_from_hdf5(
    data_folder: str,
    task_id: int,
    demo_id: int,
    args,
    sampling_rate: int,
    flush_every_n_steps: int = 500,
) -> None:
    task_name = TASK_INDICES_TO_NAMES[task_id]
    replay_dir = os.path.join(data_folder, "replayed")
    makedirs_with_mode(replay_dir)
    
    data_dir = os.path.join("mapping/dataset", f"task-{task_id:04d}", f"episode_{demo_id:08d}")  
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    seg_instance_id_dir = os.path.join(data_dir, "seg_instance_id")
    seg_instance_dir = os.path.join(data_dir, "seg_instance")
    seg_semantic_dir = os.path.join(data_dir, "seg_semantic")
    poses_dir = os.path.join(data_dir, "poses")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(seg_instance_id_dir, exist_ok=True)
    os.makedirs(seg_instance_dir, exist_ok=True)
    os.makedirs(seg_semantic_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)

    pose_files = load_pose_files(data_folder, task_id, demo_id)
    if not pose_files:
        print("Error: No pose files found. Please run dataset_reader.py first.")
        return
    print(f"Found a total of {len(pose_files)} pose files to render.")

    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb"],
            "sensor_kwargs": {
                "image_height": 128,
                "image_width": 128,
            },
        },
    }

    
    modalities = ["rgb", "depth_linear", "seg_instance_id", "seg_semantic", "seg_instance"]
    external_sensors_config = [{
        "name": "my_sensor",
        "sensor_type": "VisionSensor",
        "modalities": modalities,
        "sensor_kwargs": {
            "image_height": 512,
            "image_width": 512,
        },
        "position": [0, 0, 1.5],
        "orientation": [0, 0, 0, 1],
    }]
    
    sorted_pose_files = sorted(pose_files)
    num_poses = len(sorted_pose_files)

    # Determine where to start saving, based on sequentially named existing files
    start_save_idx = 0
    if os.path.exists(rgb_dir):
        existing_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
        if existing_files:
            try:
                last_saved_idx = max(int(os.path.splitext(f)[0]) for f in existing_files)
                start_save_idx = last_saved_idx + 1
            except ValueError:
                # No valid file names found
                start_save_idx = 0
    
    if start_save_idx > 0:
        print(f"Output directory not empty. Starting from save index {start_save_idx}.")

    # The loop for iterating through poses should start at the corresponding sampled index
    start_frame_idx = start_save_idx * sampling_rate

    if start_frame_idx >= num_poses:
        log.info(f"All {num_poses // sampling_rate} frames have been generated.")
        return

    # Get the full scene file path, following the logic in replay_obs.py
    # This is the correct way to load the scene for playback.
    task_scene_file_folder = os.path.join(
        os.path.dirname(os.path.dirname(og.__path__[0])), "joylo", "sampled_task", task_name
    )
    full_scene_file = None
    for file in os.listdir(task_scene_file_folder):
        if file.endswith(".json") and "partial_rooms" not in file:
            full_scene_file = os.path.join(task_scene_file_folder, file)
    assert full_scene_file is not None, f"No full scene file found in {task_scene_file_folder}"

    env = BehaviorDataPlaybackWrapper.create_from_hdf5(
        input_path=f"{data_folder}/2025-challenge-rawdata/task-{task_id:04d}/episode_{demo_id:08d}.hdf5",
        output_path=os.path.join(replay_dir, f"episode_{demo_id:08d}.hdf5"),
        compression={"compression": "lzf"},
        robot_obs_modalities=["proprio"],
        robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        n_render_iterations=20,
        flush_every_n_traj=1,
        flush_every_n_steps=flush_every_n_steps,
        include_robot_control=False,
        include_contacts=False,
        full_scene_file=full_scene_file,
    )
    
    num_samples = [env.input_hdf5["data"][key].attrs["num_samples"] for key in env.input_hdf5["data"].keys()]
    episode_id = num_samples.index(max(num_samples))
    log.info(f" >>> Replaying episode {episode_id}")
    
    env.playback_episode(episode_id=episode_id, record_data=False, only_first_step=True)
    robot = env.robots[0]
    robot.set_position_orientation(position=np.array([-1.0, -10.0, 2.0]))
    sensor = env._external_sensors["my_sensor"]

    image_width = sensor.image_width
    image_height = sensor.image_height
    horizontal_aperture = sensor.horizontal_aperture
    focal_length = sensor.focal_length
    vertical_aperture = horizontal_aperture * (image_height / image_width)
    
    cx = image_width / 2.0
    cy = image_height / 2.0
    fx = focal_length * image_width / horizontal_aperture
    fy = focal_length * image_height / vertical_aperture

    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    intrinsics_path = os.path.join(data_dir, "intrinsics.txt")
    np.savetxt(intrinsics_path, intrinsics)

    save_idx = start_save_idx
    
    # Correctly calculate the number of items for tqdm
    num_to_render = (num_poses - start_frame_idx + sampling_rate - 1) // sampling_rate

    cumulative_instance_id_map = {}

    with tqdm(total=num_to_render, desc="Generating samples") as pbar:
        for frame_idx in range(start_frame_idx, num_poses, sampling_rate):
            pose_file = sorted_pose_files[frame_idx]

            # 1. Load a pose from the pre-generated list
            extrinsic_cv = np.load(pose_file)
            point, orientation = get_pose_from_extrinsic(extrinsic_cv)

            # 2. Collision check (no longer needed as poses are from valid demos)

            # 3. Proceed with rendering
            robot.set_position_orientation(position=np.array([-1.0, -10.0, 2.0]))

            sensor.set_position_orientation(position=point, orientation=orientation)
            for _ in range(20):
                og.sim.render()

            cur_pos, cur_orn = sensor.get_position_orientation()

            obs, info = sensor.get_obs()

            # Merge the instance ID map from the current frame
            if "seg_instance_id" in info:
                # The keys are strings of integers. Let's convert them to ints.
                current_map = {int(k): v for k, v in info["seg_instance_id"].items()}
                cumulative_instance_id_map.update(current_map)

                # For debugging: save the cumulative map in every iteration
                mapping_path = os.path.join(data_dir, "instance_id_to_name.json")
                with open(mapping_path, 'w') as f:
                    json.dump(cumulative_instance_id_map, f, indent=4)

            rgb_image = copy.deepcopy(obs["rgb"][..., :3].cpu().numpy())
            depth_image = copy.deepcopy(obs["depth_linear"].cpu().numpy().squeeze())
            seg_instance_id_image = copy.deepcopy(obs["seg_instance_id"].cpu().numpy().squeeze())
            seg_instance_image = copy.deepcopy(obs["seg_instance"].cpu().numpy().squeeze())
            seg_semantic_image = copy.deepcopy(obs["seg_semantic"].cpu().numpy().squeeze())

            # Save the sample
            save_data(save_idx, rgb_image, depth_image, seg_instance_id_image, cur_pos, cur_orn, rgb_dir, depth_dir, seg_instance_id_dir, poses_dir, seg_semantic_image=seg_semantic_image, seg_semantic_dir=seg_semantic_dir, seg_instance_image=seg_instance_image, seg_instance_dir=seg_instance_dir)
            save_idx += 1
            pbar.update(1)
            time.sleep(0.01)

    log.info(f"Data generation complete. Saved {save_idx - start_save_idx} new samples. Total samples in dir: {save_idx}.")


def main():
    parser = argparse.ArgumentParser(description="Render images from sampled poses.")
    parser.add_argument("--data_folder", type=str, default="/home/sunghwan/workspace/omnigibson/DATASETS/behavior", help="Path to the data folder.")
    parser.add_argument("--task_id", type=int, default=21, help="Task ID.")
    parser.add_argument("--demo_id", type=int, default=210170, help="Demo ID.")
    parser.add_argument("--xy_file", type=str, default="point_cloud_xy.npy", help="Path to .npy file for xy sampling.")
    parser.add_argument("--radius", type=float, default=0.1, help="Collision check sphere radius")
    parser.add_argument("--sampling_rate", type=int, default=10, help="The rate at which to sample the frames.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    build_scene_from_hdf5(
        data_folder=args.data_folder,
        task_id=args.task_id,
        demo_id=args.demo_id,
        args=args,
        sampling_rate=args.sampling_rate,
    )

    log.info(f"Data saved to {os.path.join('mapping/dataset', f'task-{args.task_id:04d}', f'episode_{args.demo_id:08d}')}")
    
    og.shutdown()


if __name__ == "__main__":
    main()