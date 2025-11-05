import os
import json
import numpy as np
import imageio.v2 as imageio
import glob
import open3d as o3d
import copy

def get_instance_ids_for_object(metadata_path, object_name_prefix):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    ins_id_mapping_str = metadata.get("ins_id_mapping", "{}")
    ins_id_mapping = json.loads(ins_id_mapping_str)

    config_str = metadata.get("config", "{}")
    config_data = json.loads(config_str)
    inst_to_name = config_data.get("scene", {}).get("scene_file", {}).get("metadata", {}).get("task", {}).get("inst_to_name", {})
    
    target_object_names = []
    for key, name in inst_to_name.items():
        if key.startswith(object_name_prefix):
            target_object_names.append(name)
            
    if not target_object_names:
        print(f"Warning: No objects found with prefix '{object_name_prefix}'")
        return []

    print(f"Found target object names: {target_object_names}")

    target_instance_ids = []
    for instance_id, path_name in ins_id_mapping.items():
        if any(obj_name in path_name for obj_name in target_object_names):
            target_instance_ids.append(int(instance_id))
            
    print(f"Found target instance IDs: {target_instance_ids}")
    return target_instance_ids

def create_point_cloud_from_frame(depth_img, seg_img, rgb_img, intrinsics, pose, target_ids):
    height, width = depth_img.shape
    mask = np.isin(seg_img, target_ids)
    y, x = np.where(mask)
    depth_values = depth_img[y, x]

    if depth_values.size == 0:
        return o3d.geometry.PointCloud()

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x_cam = (x - cx) * depth_values / fx
    y_cam = (y - cy) * depth_values / fy
    z_cam = depth_values
    
    points_camera_frame = np.vstack((x_cam, y_cam, z_cam)).T
    colors = rgb_img[y, x] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_camera_frame)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # world <- cam
    cam_to_world = np.linalg.inv(pose)
    pcd.transform(cam_to_world)
    return pcd

def centroid(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return np.zeros(3)
    return pts.mean(axis=0)

def make_transform(R=np.eye(3), t=np.zeros(3)):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def main():
    # --- Configuration ---
    DATA_FOLDER = "/home/sunghwan/workspace/omnigibson/DATASETS/behavior"
    TASK_ID = 27
    DEMO_ID = 270010
    TARGET_OBJECT_PREFIX = "bottle__of__detergent.n.01" 

    episode_dir = os.path.join(DATA_FOLDER, "processed_data", f"task-{TASK_ID:04d}", f"episode_{DEMO_ID:08d}")
    if not os.path.isdir(episode_dir):
        print(f"Error: Episode directory not found at {episode_dir}")
        print("Please run script/dataset_reader.py first to generate the data.")
        return

    # --- Load metadata and find target instance IDs ---
    json_path = os.path.join(episode_dir, f"episode_{DEMO_ID:08d}.json")
    target_ids = get_instance_ids_for_object(json_path, TARGET_OBJECT_PREFIX)
    if not target_ids:
        print("Could not find any instance IDs for the target object. Exiting.")
        return

    # --- Load camera intrinsics ---
    intrinsics_path = os.path.join(episode_dir, "intrinsics.txt")
    intrinsics = np.loadtxt(intrinsics_path)

    # --- Process each frame ---
    rgb_files = sorted(glob.glob(os.path.join(episode_dir, 'rgb', '*.png')))
    pcds_with_info = []
    total_frames_to_process = len(rgb_files) // 20
    print(f"Processing {len(rgb_files)} frames, sampling one every 20 frames (approx. {total_frames_to_process} frames)...")

    for i, rgb_file in enumerate(rgb_files):
        if i % 20 != 0:
            continue

        frame_name = os.path.basename(rgb_file).split('.')[0]
        depth_file = os.path.join(episode_dir, 'depth', f'{frame_name}.npy')
        seg_file   = os.path.join(episode_dir, 'seg_instance_id', f'{frame_name}.npy')
        pose_file  = os.path.join(episode_dir, 'poses', f'{frame_name}.npy')

        if not all(os.path.exists(f) for f in [depth_file, seg_file, pose_file]):
            print(f"Skipping frame {frame_name}, missing data.")
            continue
            
        rgb_image  = imageio.imread(rgb_file)
        depth_image = np.load(depth_file)
        seg_image   = np.load(seg_file)
        pose        = np.load(pose_file)
        
        pcd = create_point_cloud_from_frame(depth_image, seg_image, rgb_image, intrinsics, pose, target_ids)
        if pcd.has_points():
            cam_to_world = np.linalg.inv(pose)
            pcds_with_info.append({'pcd': pcd, 'pose_cam_to_world': cam_to_world, 'frame': frame_name})
            print(f"  Found object in frame {frame_name}. Total found: {len(pcds_with_info)}")

    if len(pcds_with_info) < 2:
        print("Need at least two point clouds to perform ICP. Exiting.")
        if pcds_with_info:
            # 단일 포인트클라우드만 우선 보여줌
            o3d.visualization.draw_geometries([pcds_with_info[0]['pcd']], window_name="Object Point Clouds")
        return

    print("\nPerforming ICP between adjacent frames...")
    # 누적 object pose (첫 프레임 기준) 리스트. 첫 프레임은 항등.
    acc_T_icp_to_first = [np.eye(4)]

    # ---- 두 창에 넣을 지오메트리 컨테이너를 분리 ----
    geoms_pose = []   # 좌표축 / 포즈 궤적 전용
    geoms_pcd  = []   # 포인트클라우드 전용

    # 첫 프레임 객체 중심 (월드 좌표)
    first_pcd = pcds_with_info[0]['pcd']
    obj_origin_world = centroid(first_pcd)

    # object 좌표계 원점을 월드에 배치하는 평행이동
    T_place_origin = make_transform(t=obj_origin_world)

    # 첫 프레임의 object pose frame (항등)
    first_obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    first_obj_frame.transform(T_place_origin)  # 첫 프레임 원점에 배치
    geoms_pose.append(first_obj_frame)

    # 첫 프레임 포인트클라우드
    geoms_pcd.append(first_pcd)

    threshold = 0.02
    trans_init = np.eye(4)

    for i in range(1, len(pcds_with_info)):
        ref_info = pcds_with_info[i-1]
        cur_info = pcds_with_info[i]

        ref_pcd = ref_info['pcd']
        cur_pcd = cur_info['pcd']

        print(f"\nProcessing frame: {cur_info['frame']} against previous frame {ref_info['frame']}")

        # 현재 -> 이전 프레임으로 정합되는 변환
        reg_p2p = o3d.pipelines.registration.registration_icp(
            cur_pcd, ref_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        print("ICP Transformation (current -> previous):")
        print(reg_p2p.transformation)
        print(f"Fitness: {reg_p2p.fitness}")
        print(f"Inlier RMSE: {reg_p2p.inlier_rmse}")

        # 누적: T_i(first<-i) = T_{i-1}(first<-i-1) @ T(current i -> previous i-1)
        T_to_first = acc_T_icp_to_first[-1] @ reg_p2p.transformation
        acc_T_icp_to_first.append(T_to_first)

        # ----- 포즈 창(좌표축)용 지오메트리 -----
        obj_frame_i = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
        # 첫 프레임 원점 배치 후, 누적 ICP 적용
        T_pose_i = T_place_origin @ T_to_first
        obj_frame_i.transform(T_pose_i)
        geoms_pose.append(obj_frame_i)

        # ----- 포인트클라우드 창용 지오메트리 -----
        # 월드 좌표에 이미 있어 비교 관찰이 쉬움. 정합된 결과를 보고 싶으면 주석 해제.
        geoms_pcd.append(cur_pcd)

        # # 정합 결과(이전 프레임 기준)도 같이 보고 싶다면:
        # pcd_aligned = copy.deepcopy(cur_pcd)
        # pcd_aligned.transform(reg_p2p.transformation)
        # geoms_pcd.append(pcd_aligned)

    print("\n[1/2] Visualizing ICP pose trajectory (axes only). Press 'q' to close.")
    o3d.visualization.draw_geometries(geoms_pose, window_name="ICP Pose Trajectory")

    print("\n[2/2] Visualizing object point clouds. Press 'q' to close.")
    o3d.visualization.draw_geometries(geoms_pcd, window_name="Object Point Clouds")

if __name__ == "__main__":
    main()
