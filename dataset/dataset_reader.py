import os
import json
import torch as th
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.sensors import VisionSensor
from omnigibson.learning.utils.dataset_utils import makedirs_with_mode
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    TASK_INDICES_TO_NAMES,
)
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger
from typing import Dict
import numpy as np
import imageio
from scipy.spatial.transform import Rotation
import argparse

log = create_module_logger(module_name="replay_obs")
log.setLevel(20)

gm.RENDER_VIEWER_CAMERA = False
# gm.ENABLE_HQ_RENDERING = False
gm.HEADLESS = False

gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128
gm.ENABLE_TRANSITION_RULES = False

FLUSH_EVERY_N_STEPS = 500


def save_data(frame_idx, rgb_image, depth_image, seg_instance_id_image, flow_image, pos, orn, rgb_dir, depth_dir, seg_instance_id_dir, flow_dir, poses_dir):
    frame_filename_base = f"{frame_idx:05d}"
    imageio.imwrite(os.path.join(rgb_dir, f"{frame_filename_base}.png"), rgb_image)
    np.save(os.path.join(depth_dir, f"{frame_filename_base}.npy"), depth_image)
    np.save(os.path.join(seg_instance_id_dir, f"{frame_filename_base}.npy"), seg_instance_id_image)
    np.save(os.path.join(flow_dir, f"{frame_filename_base}.npy"), flow_image)
    
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


class BehaviorDataPlaybackWrapper(DataPlaybackWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # These will be set manually after creation
        self.cam_data_dirs = None
        self.sampling_rate = 1
        self.frame_idx = -1
        self.base_output_dir = None
        self.cumulative_instance_id_map = {}

    def reset(self, *args, **kwargs):
        self.frame_idx = -1
        self.cumulative_instance_id_map = {}
        return super().reset(*args, **kwargs)

    def _process_obs(self, obs, info):
        # Increment frame counter first
        self.frame_idx += 1

        # --- Start of logic based on replay_obs.py ---
        robot = self.env.robots[0]
        base_pose = robot.get_position_orientation()
        cam_rel_poses = []
        

        # Update and save the instance ID mapping for each camera
        if self.base_output_dir:
            if 'obs_info' in info:
                for robot_obs_info in info['obs_info'].values():
                    if not isinstance(robot_obs_info, dict):
                        continue
                    for camera_obs in robot_obs_info.values():
                        if isinstance(camera_obs, dict) and 'seg_instance_id' in camera_obs:
                            id_map = camera_obs['seg_instance_id']
                            if isinstance(id_map, dict):
                                self.cumulative_instance_id_map.update(id_map)

            mapping_path = os.path.join(self.base_output_dir, "instance_id_to_name.json")
            with open(mapping_path, 'w') as f:
                json.dump(self.cumulative_instance_id_map, f, indent=4)

        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            # Standard processing from replay_obs.py
            assert camera_name.split("::")[1] in robot.sensors, f"Camera {camera_name} not found in robot sensors"
            if f"{camera_name}::seg_semantic" in obs:
                obs.pop(f"{camera_name}::seg_semantic")
            if f"{camera_name}::seg_instance_id" in obs:
                obs[f"{camera_name}::seg_instance_id"] = obs[f"{camera_name}::seg_instance_id"].cpu()

            # Get world pose for this camera
            cam_pose = robot.sensors[camera_name.split("::")[1]].get_position_orientation()

            # Save data for this camera if dirs are set up for it and the frame is sampled
            if self.frame_idx % self.sampling_rate == 0:
                if self.cam_data_dirs and camera_name in self.cam_data_dirs:
                    log.info(f"Saving frame {self.frame_idx} for camera {camera_name}...")
                    rgb_image = obs[f"{camera_name}::rgb"][..., :3].cpu().numpy().copy()
                    depth = obs[f"{camera_name}::depth_linear"].cpu().numpy().squeeze()
                    seg = obs[f"{camera_name}::seg_instance_id"].cpu().numpy().squeeze()
                    flow = obs[f"{camera_name}::flow"].cpu().numpy().squeeze()
                    pos, orn = cam_pose
                    cam_dirs = self.cam_data_dirs[camera_name]
                    save_data(self.frame_idx, rgb_image, depth, seg, flow, pos, orn,
                              cam_dirs["rgb"], cam_dirs["depth"], cam_dirs["seg_instance_id"], cam_dirs["flow"], cam_dirs["poses"])

            # Calculate relative pose and add to list (this is saved in HDF5)
            cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))

        # This key is expected by other parts of the system, so we add it.
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        # --- End of logic based on replay_obs.py ---

        return obs

    def postprocess_traj_group(self, traj_grp):
        log.info(f"Postprocessing trajectory group {traj_grp.name}")
        traj_grp.attrs["robot_type"] = "R1Pro"
        traj_grp.attrs["task_obs_keys"] = self.env.task.low_dim_obs_keys
        traj_grp.attrs["ins_id_mapping"] = json.dumps(VisionSensor.INSTANCE_ID_REGISTRY)

        camera_names = set(ROBOT_CAMERA_NAMES["R1Pro"].values())
        for name in self.env.robots[0].sensors:
            if f"robot_r1::{name}" in camera_names:
                unique_ins_ids = set()
                for i in range(0, traj_grp["obs"][f"robot_r1::{name}::seg_instance_id"].shape[0], FLUSH_EVERY_N_STEPS):
                    unique_ins_ids.update(
                        th.unique(
                            th.from_numpy(
                                traj_grp["obs"][f"robot_r1::{name}::seg_instance_id"][i : i + FLUSH_EVERY_N_STEPS]
                            )
                        )
                        .to(th.uint32)
                        .tolist()
                    )
                traj_grp.attrs[f"robot_r1::{name}::unique_ins_ids"] = list(unique_ins_ids)
        log.info(f"Postprocessing trajectory group {traj_grp.name} done")


def replay_hdf5_file(
    data_folder: str,
    task_id: int,
    demo_id: int,
    camera_names: Dict[str, str] = ROBOT_CAMERA_NAMES["R1Pro"],
    flush_every_n_steps: int = 500,
    sampling_rate: int = 1,
    horizontal_aperture: float = 20.995,
    focal_length: float = 17.0,
    image_height: int = 336,
    image_width: int = 336,
) -> None:
    task_name = TASK_INDICES_TO_NAMES[task_id]
    replay_dir = os.path.join(data_folder, "replayed")
    makedirs_with_mode(replay_dir)
    
    base_output_dir = os.path.join(data_folder, "processed_data", f"task-{task_id:04d}", f"episode_{demo_id:08d}")
    cam_data_dirs = {}
    for cam_key, cam_name in camera_names.items():
        output_data_dir = os.path.join(base_output_dir, cam_key)
        rgb_dir = os.path.join(output_data_dir, "rgb")
        depth_dir = os.path.join(output_data_dir, "depth")
        seg_instance_id_dir = os.path.join(output_data_dir, "seg_instance_id")
        poses_dir = os.path.join(output_data_dir, "poses")
        flow_dir = os.path.join(output_data_dir, "flow")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(seg_instance_id_dir, exist_ok=True)
        os.makedirs(poses_dir, exist_ok=True)
        os.makedirs(flow_dir, exist_ok=True)

        cam_data_dirs[cam_name] = {
            "rgb": rgb_dir,
            "depth": depth_dir,
            "seg_instance_id": seg_instance_id_dir,
            "flow": flow_dir,
            "poses": poses_dir,
        }

    modalities = ["rgb", "depth_linear", "seg_instance_id", "flow"]
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": modalities,
            "sensor_kwargs": {
                "image_height": image_height,
                "image_width": image_width,
            },
        },
    }

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
    
    # --- Configure cameras and save intrinsics ---
    for cam_key, cam_name in camera_names.items():
        camera_sensor = env.robots[0].sensors[cam_name.split("::")[1]]

        # Set camera parameters for all cameras
        camera_sensor.horizontal_aperture = horizontal_aperture
        camera_sensor.focal_length = focal_length

        # Customize camera properties if needed (e.g., for higher resolution)
        if cam_key == "head":
            camera_sensor.image_height = image_height
            camera_sensor.image_width = image_width

        # This is needed to update the observation space after changing sensor properties
        env.load_observation_space()

        # Save intrinsics by calculating from sensor params
        image_width = camera_sensor.image_width
        image_height = camera_sensor.image_height
        horizontal_aperture = camera_sensor.horizontal_aperture
        focal_length = camera_sensor.focal_length
        vertical_aperture = horizontal_aperture * (image_height / image_width)
        
        cx = image_width / 2.0
        cy = image_height / 2.0
        fx = focal_length * image_width / horizontal_aperture
        fy = focal_length * image_height / vertical_aperture

        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        output_data_dir = os.path.join(base_output_dir, cam_key)
        intrinsics_path = os.path.join(output_data_dir, "intrinsics.txt")
        np.savetxt(intrinsics_path, intrinsics)
        log.info(f"Saved {cam_key} camera intrinsics to {intrinsics_path}")

    # Manually set the save directories and sampling rate on the wrapper instance.
    env.cam_data_dirs = cam_data_dirs
    env.sampling_rate = sampling_rate
    env.base_output_dir = base_output_dir
    
    # Use the original playback_episode method. Data saving is now handled in _process_obs
    env.playback_episode(episode_id=episode_id, record_data=True)

    log.info("Playback complete. Saving data...")
    env.save_data()
    log.info(f"Successfully processed episode_{demo_id:08d}")
    return episode_id

def main():
    
    parser = argparse.ArgumentParser(description="Replay HDF5 demonstration files")
    parser.add_argument("--data_folder", type=str, default="/home/sunghwan/workspace/omnigibson/DATASETS/behavior", help="Path to the data folder")
    parser.add_argument("--task_id", type=int, default=21, help="Task ID to replay")
    parser.add_argument("--demo_id", type=int, default=210170, help="Demo ID to replay")
    parser.add_argument("--sampling_rate", type=int, default=1, help="Sampling rate for data extraction")
    parser.add_argument("--horizontal_aperture", type=float, default=20.995, help="Horizontal aperture for the camera.")
    parser.add_argument("--focal_length", type=float, default=17.0, help="Focal length for the camera.")
    parser.add_argument("--image_height", type=int, default=512, help="Image height for the head camera.")
    parser.add_argument("--image_width", type=int, default=512, help="Image width for the head camera.")
    parser.add_argument("--cameras", nargs='+', default=["head", "left_wrist", "right_wrist"], help="Cameras to replay (e.g., head left_wrist)")
    
    args = parser.parse_args()
    
    DATA_FOLDER = args.data_folder
    TASK_ID = args.task_id
    DEMO_ID = args.demo_id
    SAMPLING_RATE = args.sampling_rate
    
    # Filter camera_names based on user input
    all_camera_names = ROBOT_CAMERA_NAMES["R1Pro"]
    selected_camera_names = {key: val for key, val in all_camera_names.items() if key in args.cameras}
    if not selected_camera_names:
        log.warning(f"No valid cameras selected from {args.cameras}. Using all cameras.")
        selected_camera_names = all_camera_names

    replay_hdf5_file(
        data_folder=DATA_FOLDER,
        task_id=TASK_ID,
        demo_id=DEMO_ID,
        camera_names=selected_camera_names,
        sampling_rate=SAMPLING_RATE,
        horizontal_aperture=args.horizontal_aperture,
        focal_length=args.focal_length,
        image_height=args.image_height,
        image_width=args.image_width,
    )

    log.info(f"Data saved to {os.path.join(DATA_FOLDER, 'processed_data', f'task-{TASK_ID:04d}', f'episode_{DEMO_ID:08d}')}")
    
    og.shutdown()


if __name__ == "__main__":
    main()