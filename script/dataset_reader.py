import os
import json
import h5py
import torch as th
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.sensors import VisionSensor
from omnigibson.learning.utils.dataset_utils import makedirs_with_mode
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    PROPRIOCEPTION_INDICES,
    TASK_INDICES_TO_NAMES,
)
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger
from typing import Dict
import numpy as np
import imageio
from scipy.spatial.transform import Rotation
import math


# 로거 설정
log = create_module_logger(module_name="replay_obs")
log.setLevel(20)

# 뷰어 비활성화
gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

FLUSH_EVERY_N_STEPS = 500


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
    generate_rgbd: bool = False,
    generate_seg: bool = False,
    flush_every_n_steps: int = 500,
) -> None:
    task_name = TASK_INDICES_TO_NAMES[task_id]
    replay_dir = os.path.join(data_folder, "replayed")
    makedirs_with_mode(replay_dir)
    
    output_data_dir = os.path.join(data_folder, "processed_data", f"task-{task_id:04d}", f"episode_{demo_id:08d}")
    rgb_dir = os.path.join(output_data_dir, "rgb")
    depth_dir = os.path.join(output_data_dir, "depth")
    seg_instance_id_dir = os.path.join(output_data_dir, "seg_instance_id")
    poses_dir = os.path.join(output_data_dir, "poses")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(seg_instance_id_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)


    gm.ENABLE_TRANSITION_RULES = False

    modalities = []
    if generate_rgbd:
        modalities += ["rgb", "depth_linear"]
    if generate_seg:
        modalities += ["seg_instance_id"]
        
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": modalities,
            "sensor_kwargs": {
                "image_height": 512,
                "image_width": 512,
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
        n_render_iterations=3,
        flush_every_n_traj=1,
        flush_every_n_steps=flush_every_n_steps,
        include_robot_control=False,
        include_contacts=False,
        full_scene_file=full_scene_file,
    )
    
    if generate_rgbd:
        head_camera_config = env.robots[0].sensors["robot_r1:zed_link:Camera:0"]
        head_camera_config.horizontal_aperture = 40.0
        head_camera_config.image_height = 512
        head_camera_config.image_width = 512
        env.load_observation_space()
        
        # Save intrinsics by calculating from sensor params
        image_width = head_camera_config.image_width
        image_height = head_camera_config.image_height
        horizontal_aperture = head_camera_config.horizontal_aperture
        focal_length = head_camera_config.focal_length
        vertical_aperture = horizontal_aperture * (image_height / image_width)
        
        cx = image_width / 2.0
        cy = image_height / 2.0
        fx = focal_length * image_width / horizontal_aperture
        fy = focal_length * image_height / vertical_aperture

        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics_path = os.path.join(output_data_dir, "intrinsics.txt")
        np.savetxt(intrinsics_path, intrinsics)

    num_samples = [env.input_hdf5["data"][key].attrs["num_samples"] for key in env.input_hdf5["data"].keys()]
    episode_id = num_samples.index(max(num_samples))
    log.info(f" >>> Replaying episode {episode_id}")

    # Manually set the save directories on the wrapper instance after it's created.
    # This is the bridge that allows our _process_obs hook to save data.
    env.rgb_dir = rgb_dir
    env.depth_dir = depth_dir
    env.seg_instance_id_dir = seg_instance_id_dir
    env.poses_dir = poses_dir
    env.head_cam_name = ROBOT_CAMERA_NAMES["R1Pro"]["head"]
    
    # Use the original playback_episode method. Data saving is now handled in _process_obs
    env.playback_episode(episode_id=episode_id, record_data=True)

    log.info("Playback complete. Saving data...")
    env.save_data()
    log.info(f"Successfully processed episode_{demo_id:08d}")
    return episode_id

def main():
    # 데이터 경로와 replay할 task/demo ID를 설정해주세요.
    # DATA_FOLDER는 .../2025-challenge-rawdata 와 .../processed_data를 포함하는 상위 디렉토리여야 합니다.
    DATA_FOLDER = "/home/sunghwan/workspace/omnigibson/DATASETS/behavior"
    TASK_ID = 27
    DEMO_ID = 270010
    
    # 원하시는 데이터(rgb, depth, segmentation)를 얻기 위해 replay를 수행합니다.
    # camera pose는 replay된 hdf5 파일에 저장됩니다.
    replay_hdf5_file(
        data_folder=DATA_FOLDER,
        task_id=TASK_ID,
        demo_id=DEMO_ID,
        generate_rgbd=True,
        generate_seg=True,
    )

    log.info(f"Data saved to {os.path.join(DATA_FOLDER, 'processed_data', f'task-{TASK_ID:04d}', f'episode_{DEMO_ID:08d}')}")
    
    og.shutdown()


if __name__ == "__main__":
    main()