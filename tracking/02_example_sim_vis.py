import open3d as o3d
import numpy as np
import os
import imageio.v2 as imageio
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Optional
import torch.nn as nn
import sys
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mapping.mapping_lib.implicit_decoder import ImplicitDecoder
from mapping.mapping_lib.voxel_hash_table import VoxelHashTable


# ==============================================================================================
#  Model Wrappers & Helpers (from mapping/latent_map.py)
# ==============================================================================================

# DINOv3 wrapper class
class DINOv3Wrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = backbone.embed_dim

        # DINOv3 normalization
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    
    @torch.no_grad()
    def _forward_dino_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-patch token embeddings without the [CLS] token.
        Shape: (B, N, C), where N = (H/14)*(W/14), C = embed_dim.
        """
        x, (H, W) = self.backbone.prepare_tokens_with_masks(x)

        for blk in self.backbone.blocks:
            if hasattr(self.backbone, "rope_embed") and self.backbone.rope_embed is not None:
                rope_sincos = self.backbone.rope_embed(H=H, W=W)
            else:
                raise ValueError("Rope embedding not found in DINOv3")
            x = blk(x, rope_sincos)
        x = self.backbone.norm(x)  # (B, 1 + N, C)
        x = x[:, 5:, :]  # drop CLS and storage tokens

        return x

    def forward(self, images_bchw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_bchw: Float tensor in [0, 1], shape (B, 3, H, W)

        Returns:
            fmap: (B, C, Hf, Wf) where C = feature_dim and Hf = H//16, Wf = W//16
        """
        if images_bchw.dtype != torch.float32:
            images_bchw = images_bchw.float()
        # Normalize per DINOv3 recipe
        images_bchw = self.normalize(images_bchw)

        B, _, H, W = images_bchw.shape
        with torch.no_grad():
            tokens = self._forward_dino_tokens(images_bchw)  # (B, N, C)

        C = self.feature_dim
        patch_size = 16 # Varies by DINO model, vith16plus is 16
        Hf, Wf = H // patch_size, W // patch_size
        fmap = tokens.permute(0, 2, 1).reshape(B, C, Hf, Wf).contiguous()

        return fmap

def get_3d_coordinates_from_depth(depth, E_cv, fx, fy, cx, cy, device='cpu'):
    """
    Calculates 3D coordinates in world frame from a depth map.
    depth: (B, H, W)
    E_cv: (B, 3, 4) camera-to-world extrinsic matrix
    """
    B, H, W = depth.shape
    
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    z_cam = depth.reshape(B, -1)
    x_cam = (u.reshape(-1) - cx) * z_cam / fx
    y_cam = (v.reshape(-1) - cy) * z_cam / fy
    
    # (B, 3, H*W)
    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)
    
    R_c2w = E_cv[:, :3, :3]
    t_c2w = E_cv[:, :3, 3].unsqueeze(-1)
    
    # (B, 3, H*W)
    pts_world = torch.bmm(R_c2w, pts_cam) + t_c2w
    
    return pts_world.permute(0, 2, 1).reshape(B, H, W, 3)

def main():
    """
    Visualizes an aggregated point cloud from an RGBD dataset with DINOv3 features.
    """
    parser = argparse.ArgumentParser(description="Visualize a point cloud from an RGBD dataset with DINOv3 features.")
    parser.add_argument("--data_dir", type=str, default="DATASETS/behavior/processed_data/task-0021/episode_00210170/", help="Path to the dataset directory.")
    parser.add_argument("--camera_name", type=str, default="head", choices=["head", "left_wrist", "right_wrist"], help="Name of the camera to use.")
    parser.add_argument("--start_frame", type=int, default=970, help="The starting frame index.")
    parser.add_argument("--end_frame", type=int, default=1070, help="The ending frame index.")
    parser.add_argument("--step", type=int, default=5, help="The step between frames.")
    parser.add_argument("--dino_weights_path", type=str, default='dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth', help="Path to DINOv3 weights file (e.g., dinov3_vith16plus.pt).")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==============================================================================================
    # Load Latent Map Model
    # ==============================================================================================
    # Load config
    config_path = "mapping/config.yaml"
    print(f"[INIT] Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Define paths based on config and latest run
    # NOTE: This assumes the latest run is the one to be used.
    run_dir = "mapping/map_output/task-0021/runs/task-0021_20251113-193630"
    decoder_path = os.path.join(run_dir, "decoder.pt")
    
    # Try to determine env_name from the data_dir arg
    env_name = Path(args.data_dir).name
    grid_path = os.path.join(run_dir, env_name, "grid_dense.pt")
    
    if not os.path.exists(grid_path):
        print(f"Warning: Grid not found at {grid_path}.")
        # Fallback to the env name found in the runs directory if the one from args doesn't exist
        found_envs = [d.name for d in Path(run_dir).iterdir() if d.is_dir()]
        if found_envs:
            env_name = found_envs[0]
            grid_path = os.path.join(run_dir, env_name, "grid_dense.pt")
            print(f"Found and using grid for env: {env_name} at {grid_path}")
        else:
            print(f"Error: No environment grid found in {run_dir}. Exiting.")
            return

    # Assuming dino model as per default config
    feature_dim = 1280 # DINOv3-ViT-H/16plus feature dimension

    # Load Decoder
    print(f"[INIT] Loading decoder from {decoder_path}")
    decoder_config = config['decoder']
    grid_config = config['grid']
    decoder = ImplicitDecoder(
        voxel_feature_dim=grid_config['feature_dim'] * grid_config['levels'],
        hidden_dim=decoder_config['hidden_dim'],
        output_dim=feature_dim,
    ).to(DEVICE)
    decoder.load_state_dict(torch.load(decoder_path, map_location=DEVICE))
    decoder.eval()
    print("[INIT] Decoder loaded.")

    # Load Grid
    print(f"[INIT] Loading dense grid from {grid_path}")
    grid_config = config['grid']
    grid = VoxelHashTable(
        resolution=grid_config['resolution'],
        num_levels=grid_config['levels'],
        level_scale=grid_config['level_scale'],
        feature_dim=grid_config['feature_dim'],
        hash_table_size=grid_config['hash_table_size'],
        scene_bound_min=tuple(config['scene_min']),
        scene_bound_max=tuple(config['scene_max']),
        device=DEVICE,
        mode="train"
    )
    
    chk = torch.load(grid_path, map_location=DEVICE)
    grid.load_state_dict(chk['state_dict'])
    grid.eval()
    print("[INIT] Grid loaded.")

    # --------------------------------------------------------------------------- #
    #  Load DINOv3 Model
    # --------------------------------------------------------------------------- #
    print(f"[INIT] Loading DINOv3 model...")
    try:
        backbone = torch.hub.load('dinov3', 'dinov3_vith16plus', source='local', weights=args.dino_weights_path, force_reload=True)
        model = DINOv3Wrapper(backbone).to(DEVICE).eval()
        feature_dim = model.feature_dim
        patch_size = 16
        print(f"[INIT] Loaded DINOv3 model with feature dimension {feature_dim}.")
    except Exception as e:
        print(f"Error loading DINOv3 model: {e}")
        print("Please ensure the 'dinov3' repository is in your python path and the weights path is correct.")
        return
    
    # --------------------------------------------------------------------------- #
    #  Load Data
    # --------------------------------------------------------------------------- #
    data_dir = os.path.join(args.data_dir, args.camera_name)
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
        
    fx_orig, fy_orig, cx_orig, cy_orig = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # Get file lists
    try:
        rgb_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.png')])
        depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.npy')])
        pose_files = sorted([os.path.join(poses_dir, f) for f in os.listdir(poses_dir) if f.endswith('.npy')])
    except FileNotFoundError as e:
        print(f"Error: Data directory not found or incomplete. {e}")
        return

    if args.end_frame != -1:
        rgb_files = rgb_files[:args.end_frame]
        depth_files = depth_files[:args.end_frame]
        pose_files = pose_files[:args.end_frame]

    if len(rgb_files) == 0 or len(depth_files) == 0 or len(pose_files) == 0:
        print("Error: No rgb/depth/pose files found.")
        return

    all_coords_list = []
    all_feats_list = []

    end_frame = len(rgb_files)
    print(f"Processing frames from {args.start_frame} to {end_frame} (step={args.step})")
    for i in range(args.start_frame, end_frame, args.step):
        print(f"Processing frame {i+1}/{len(rgb_files)}: {os.path.basename(rgb_files[i])}")
        
        # Load data for the current frame
        rgb_image = imageio.imread(rgb_files[i])
        depth_image = np.load(depth_files[i])
        pose_world_to_cam = np.load(pose_files[i])
        
        h_orig, w_orig, _ = rgb_image.shape
        
        # --------------------------------------------------------------------------- #
        #  Feature Extraction & 3D Point Calculation
        # --------------------------------------------------------------------------- #
        
        # 1. Prepare image for DINOv3 using its original size
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        img_tensor_orig = rgb_tensor.unsqueeze(0).to(DEVICE)
        
        # 2. Extract features
        with torch.no_grad():
            vis_feat = model(img_tensor_orig) # (1, C, Hf, Wf)
        
        B, C, Hf, Wf = vis_feat.shape
        feats_valid_frame = vis_feat.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        
        # 3. Prepare depth and pose for point calculation
        depth_t = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).float()
        depth_t_resized = F.interpolate(depth_t, (Hf, Wf), mode="nearest-exact").squeeze(0).to(DEVICE)
        
        pose_cam_to_world = np.linalg.inv(pose_world_to_cam)
        extrinsic_t = torch.from_numpy(pose_cam_to_world[:3, :]).unsqueeze(0).float().to(DEVICE)

        # 4. Adjust intrinsics for feature map resolution
        fx = fx_orig * (Wf / w_orig)
        fy = fy_orig * (Hf / h_orig)
        cx = cx_orig * (Wf / w_orig)
        cy = cy_orig * (Hf / h_orig)

        # 5. Get 3D coordinates for each feature vector
        coords_world = get_3d_coordinates_from_depth(depth_t_resized, extrinsic_t, fx, fy, cx, cy, device=DEVICE)
        
        # ==============================================================================================
        #  Query Grid and Calculate Cosine Similarity
        # ==============================================================================================
        # 1. Reshape coordinates for grid query
        coords_for_grid = coords_world.reshape(-1, 3) # (Hf*Wf, 3)

        # 2. Query grid and decode features
        with torch.no_grad():
            # Use mark_accessed=False as we are not training
            grid_feat_raw = grid.query_voxel_feature(coords_for_grid, mark_accessed=False)
            grid_feat_decoded = decoder(grid_feat_raw) # (Hf*Wf, C)

        # 3. Reshape DINO features to match
        dino_feat = vis_feat.permute(0, 2, 3, 1).reshape(-1, C) # (Hf*Wf, C)

        # 4. Calculate cosine similarity
        cosine_sim = F.cosine_similarity(grid_feat_decoded, dino_feat, dim=1) # (Hf*Wf,)
        sim_map = cosine_sim.reshape(Hf, Wf).cpu().numpy()

        # 5. Visualize the results for the first frame
        print(f"Visualizing similarity map for frame {i}")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_image)
        axes[0].set_title("RGB Image")
        axes[0].axis('off')

        im = axes[1].imshow(sim_map, cmap='viridis')
        axes[1].set_title("Cosine Similarity with Grid Features")
        axes[1].axis('off')
        fig.colorbar(im, ax=axes[1])

        axes[2].imshow(depth_image, cmap='gray')
        axes[2].set_title("Depth Image")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        # We only visualize for the first frame in this example, then break.
        # Remove the 'break' to process all frames.
        # break
        # ==============================================================================================

        coords_valid_frame = coords_world.reshape(-1, 3).cpu().numpy()

        # 6. Filter out points with no depth
        valid_depth_mask = depth_t_resized.reshape(-1).cpu().numpy() > 0.01
        coords_valid_frame = coords_valid_frame[valid_depth_mask]
        feats_valid_frame = feats_valid_frame[valid_depth_mask]

        if coords_valid_frame.shape[0] > 0:
            all_coords_list.append(coords_valid_frame)
            all_feats_list.append(feats_valid_frame)

    if not all_coords_list:
        print("No valid points found across all frames. Exiting.")
        return
        
    # Combine all points and features
    combined_coords = np.concatenate(all_coords_list, axis=0)
    combined_feats = np.concatenate(all_feats_list, axis=0)

    # --------------------------------------------------------------------------- #
    #  PCA and Visualization
    # --------------------------------------------------------------------------- #
    print(f"\n[VIS] Running PCA on {combined_feats.shape[0]} total voxel features...")
    pca = PCA(n_components=3)
    feats_pca = pca.fit_transform(combined_feats)
    scaler = MinMaxScaler()
    feats_pca_norm = scaler.fit_transform(feats_pca)

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_coords)
    pcd.colors = o3d.utility.Vector3dVector(feats_pca_norm)
    
    print("Visualizing the combined point cloud with DINOv3 PCA features...")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
