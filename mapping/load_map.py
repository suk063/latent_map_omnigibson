import os, time, argparse, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import viser
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import yaml

# local modules
from mapping_lib.utils import get_3d_coordinates
from mapping_lib.voxel_hash_table import VoxelHashTable
from mapping_lib.implicit_decoder import ImplicitDecoder


# ==============================================================================================
# PCA Visualization Function
# ==============================================================================================
def run_pca_visualization(server, grid, decoder, coords_list, env_name, device):
    """Runs PCA on voxel features and sends them to the Viser server."""
    print(f"\n[VIS] Running PCA on voxel features for {env_name}...")

    vertices_vis = torch.cat(coords_list, dim=0).numpy()

    max_points_for_vis = 5000000
    if vertices_vis.shape[0] > max_points_for_vis:
        print(f"[VIS] Downsampling from {vertices_vis.shape[0]} to {max_points_for_vis} points for PCA visualization.")
        indices = np.random.choice(vertices_vis.shape[0], max_points_for_vis, replace=False)
        vertices_vis = vertices_vis[indices]

    coords_t = torch.from_numpy(vertices_vis).to(device)
    batch_size = 50000
    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, coords_t.shape[0], batch_size), desc="[VIS] Processing features"):
            batch_coords = coords_t[i:i+batch_size]
            voxel_feat = grid.query_voxel_feature(batch_coords)
            feats_t = decoder(voxel_feat)
            all_feats.append(feats_t.cpu())

    feats_np = torch.cat(all_feats, dim=0).numpy()
    pca = PCA(n_components=3)
    feats_pca = pca.fit_transform(feats_np)
    scaler = MinMaxScaler()
    feats_pca_norm = scaler.fit_transform(feats_pca)

    z_threshold = 2.5
    vis_mask = vertices_vis[:, 2] <= z_threshold
    filtered_vertices = vertices_vis[vis_mask]
    filtered_colors = feats_pca_norm[vis_mask]
    
    # Center the point cloud (x, y) for visualization
    x_center = (config['scene_min'][0] + config['scene_max'][0]) / 2.0
    y_center = (config['scene_min'][1] + config['scene_max'][1]) / 2.0
    
    centered_vertices = filtered_vertices.copy()
    centered_vertices[:, 0] -= x_center
    centered_vertices[:, 1] -= y_center

    print(f"[VIS] Updating PCA visualization for {env_name} to Viser.")
    server.scene.add_point_cloud(
        name=f"/pca/{env_name}",
        points=centered_vertices,
        colors=(filtered_colors * 255).astype(np.uint8),
        point_size=0.01
    )


def main():
    parser = argparse.ArgumentParser(description="Load and visualize a trained map.")
    parser.add_argument(
        "--config",
        type=str,
        default="mapping/config.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    global config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SCENE_MIN = tuple(config['scene_min'])
    SCENE_MAX = tuple(config['scene_max'])

    server = viser.ViserServer(host="0.0.0.0", port=8080)
    
    model_config = config['clip_model']
    feature_dim = model_config['feature_dim']

    grid_config = config['grid']
    decoder_config = config['decoder']
    
    decoder = ImplicitDecoder(
        voxel_feature_dim=grid_config['feature_dim'] * grid_config['levels'],
        hidden_dim=decoder_config['hidden_dim'],
        output_dim=feature_dim,
    ).to(DEVICE)

    load_path = Path(config['load_run_dir'])
    
    # Load decoder weights
    decoder_path = load_path / "decoder.pt"
    state_dict = torch.load(decoder_path, map_location=DEVICE)
    if 'model' in state_dict:
        decoder.load_state_dict(state_dict['model'])
    else:
        decoder.load_state_dict(state_dict)
    print(f"[LOAD] Loaded decoder from {decoder_path}")

    print("\n[LOAD] Loading pre-trained grid for visualization...")
    grid_path = load_path / "grid.pt"
    grid_data = torch.load(grid_path, map_location=DEVICE)
    grid = VoxelHashTable(
        resolution=grid_config['resolution'],
        num_levels=grid_config['levels'],
        level_scale=grid_config['level_scale'],
        feature_dim=grid_config['feature_dim'],
        hash_table_size=grid_config['hash_table_size'],
        scene_bound_min=SCENE_MIN,
        scene_bound_max=SCENE_MAX,
        device=DEVICE,
        mode="train",
        sparse_data=grid_data
    )
    grid.eval()
    print(f"[LOAD] Loaded grid from {grid_path}")

    # Load coordinates for visualization
    dataset_path = Path(config['dataset_dir'])
    env_name = dataset_path.name
    agg_coords = []
    
    print("\n[VIS] Loading data for visualization...")
    rgb_files = sorted(list((dataset_path / "rgb").glob("*.png")))
    depth_files = sorted(list((dataset_path / "depth").glob("*.npy")))
    
    if config['num_images'] > 0:
        num_images = config['num_images']
        rgb_files = rgb_files[:num_images]
        depth_files = depth_files[:num_images]
        print(f"--- Using the first {num_images} images for visualization. ---")

    # Load intrinsics
    K = np.loadtxt(dataset_path / "intrinsics.txt")
    
    image_size = model_config['image_size']
    patch_size = model_config['patch_size']
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    fx *= (image_size / 512)
    fy *= (image_size / 512)
    cx *= (image_size / 512)
    cy *= (image_size / 512)

    # Load poses
    pose_files = sorted(glob.glob(os.path.join(config['poses_dir'], "*.npy")))
    poses_4x4 = [np.load(f) for f in pose_files]
    cam_to_world_poses = np.array([np.linalg.inv(p_4x4) for p_4x4 in poses_4x4])

    feat_h = image_size // patch_size
    for i, depth_path in enumerate(tqdm(depth_files, desc=f"Loading {env_name}")):
        depth_np = np.load(str(depth_path))
        if np.max(depth_np) == 0:
            continue
        
        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()
        depth_t = F.interpolate(depth_t, (feat_h, feat_h), mode="nearest-exact").squeeze()
        
        extrinsic_t = torch.from_numpy(cam_to_world_poses[i][:3, :]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        depth_t = depth_t.unsqueeze(0).to(DEVICE)
        
        coords_world, _ = get_3d_coordinates(
            depth_t, extrinsic_t,
            fx=fx, fy=fy, cx=cx, cy=cy,
            original_size=image_size,
        )
        
        coords_valid = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)
        
        in_x = (coords_valid[:,0] > SCENE_MIN[0]) & (coords_valid[:,0] < SCENE_MAX[0])
        in_y = (coords_valid[:,1] > SCENE_MIN[1]) & (coords_valid[:,1] < SCENE_MAX[1])
        in_z = (coords_valid[:,2] > SCENE_MIN[2]) & (coords_valid[:,2] < SCENE_MAX[2])
        in_bounds = in_x & in_y & in_z
        
        valid_depth = depth_t.reshape(-1) >= 0.01
        in_bounds &= valid_depth
        
        if in_bounds.sum() > 0:
            agg_coords.append(coords_valid[in_bounds].cpu())
    
    print(f"[VIS] Loaded coordinates for visualization.")
    
    run_pca_visualization(server, grid, decoder, agg_coords, env_name, DEVICE)

    print("\n[VIS] Viser server is running. Open the link in your browser.")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
