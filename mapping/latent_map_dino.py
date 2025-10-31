import os, time, argparse, glob
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import viser
import random
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm   
from torch.utils.tensorboard import SummaryWriter
# local modules
from mapping_lib.utils import get_3d_coordinates
from mapping_lib.voxel_hash_table import VoxelHashTable
from mapping_lib.implicit_decoder import ImplicitDecoder
from torchvision import transforms

WEIGHT_PATH = 'https://dinov3.llamameta.net/dinov3_vith16plus/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidmJ5bmdvajZodGpsbm4wNWxrd2k2NmphIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTk5NTEzNzl9fX1dfQ__&Signature=BBa0wwRRMwQKyPOqlOWi6VBJSyCEK79q9aCq%7E-MvaWPl8b%7EkQHIlzOUmKzb%7EwD1KRBgugXxGQn13x3AlV3IZnc2XVuBPmWp7eGO5ymhH6LdWgeAuLk%7Ei0bUeqr3AOWlzD3x0rKBxTDDIf-DQv0gBmSRt0IJ4edaTTcfUyDdQiRnYB6eJ2B7gz7JrnzoXtRqP5GzrtOU4nghQ3-p2ckP6zuHMUgzt4keWZoF3kekvTYqIMjmwzizDI5qm-7%7E4vYetRXpxbghCwiqiKiriYzJqm0eZys0NdW8dFfTCsHHfYd9bqE-etNrQcQffUOfUcgrqvTF7B61f5vonsQzvaznn1g__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1788493205366156'

# DINOv3 wrapper class
class DINOv3Wrapper(torch.nn.Module):
    def __init__(self, backbone, output_dim=768):
        super().__init__()
        self.backbone = backbone
        self.dino_output_dim = backbone.embed_dim  # 1024 for vit_h16plus
        self.output_dim = output_dim

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
            fmap: (B, C, Hf, Wf) where C = output_dim and Hf = H//16, Wf = W//16
        """
        if images_bchw.dtype != torch.float32:
            images_bchw = images_bchw.float()
        # Normalize per DINOv3 recipe
        images_bchw = self.normalize(images_bchw)

        B, _, H, W = images_bchw.shape
        with torch.no_grad():
            tokens = self._forward_dino_tokens(images_bchw)  # (B, N, C)

        C = self.dino_output_dim
        Hf, Wf = H // 16, W // 16
        fmap = tokens.permute(0, 2, 1).reshape(B, C, Hf, Wf).contiguous()

        return fmap

transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
])

# ==============================================================================================
# PCA Visualization Function
# ==============================================================================================
def run_pca_visualization(server, grid, decoder, coords_list, env_name, epoch_label, device):
    """Runs PCA on voxel features and sends them to the Viser server."""
    if not coords_list:
        print(f"[VIS] No coordinates to visualize for epoch {epoch_label}.")
        return

    print(f"\n[VIS] Running PCA on voxel features for {env_name} (epoch: {epoch_label})...")

    vertices_vis = torch.cat(coords_list, dim=0).numpy()

    max_points_for_vis = 2000000
    if vertices_vis.shape[0] > max_points_for_vis:
        print(f"[VIS] Downsampling from {vertices_vis.shape[0]} to {max_points_for_vis} points for PCA visualization.")
        indices = np.random.choice(vertices_vis.shape[0], max_points_for_vis, replace=False)
        vertices_vis = vertices_vis[indices]

    if vertices_vis.shape[0] == 0:
        print("[VIS] No vertices to visualize; skipping PCA visualization.")
        return

    coords_t = torch.from_numpy(vertices_vis).to(device)
    with torch.no_grad():
        voxel_feat = grid.query_voxel_feature(coords_t)
        feats_t = decoder(voxel_feat)

    feats_np = feats_t.cpu().numpy()
    pca = PCA(n_components=3)
    feats_pca = pca.fit_transform(feats_np)
    scaler = MinMaxScaler()
    feats_pca_norm = scaler.fit_transform(feats_pca)

    z_threshold = 2.0
    vis_mask = vertices_vis[:, 2] <= z_threshold
    filtered_vertices = vertices_vis[vis_mask]
    filtered_colors = feats_pca_norm[vis_mask]

    print(f"[VIS] Updating PCA visualization for {env_name} (epoch: {epoch_label}) to Viser.")
    server.add_point_cloud(
        name=f"/pca/{env_name}",
        points=filtered_vertices,
        colors=(filtered_colors * 255).astype(np.uint8),
        point_size=0.01
    )

# --------------------------------------------------------------------------- #
#  Dataset Class                                                              #
# --------------------------------------------------------------------------- #
class SingleEnvDataset(Dataset):
    def __init__(self, samples, poses_dir, transform_func):
        self.transform_func = transform_func
        print("\nLoading and processing poses...")
        
        # Load and process poses
        pose_files = sorted(glob.glob(os.path.join(poses_dir, "*.npy")))
        if not pose_files:
            raise FileNotFoundError(f"No .npy pose files found in '{poses_dir}'")
        poses_4x4 = [np.load(f) for f in pose_files]

        # The poses are world-to-camera, but we need camera-to-world to transform points from camera to world frame.
        cam_to_world_poses_list = []
        for p_4x4 in poses_4x4:
            c2w = np.linalg.inv(p_4x4)
            cam_to_world_poses_list.append(c2w)
        self.cam_to_world_poses = np.array(cam_to_world_poses_list)

        img_h = 256  # New image height
        patch_size = 16  # DINOv3 patch size
        self.feat_h = img_h // patch_size

        # Filter for valid samples without loading images into memory
        print("Validating dataset samples...")
        valid_samples = []
        for rgb_path, depth_path, pose_idx in tqdm(samples, desc="Validating"):
            if not os.path.exists(str(rgb_path)):
                continue
            try:
                # Check depth map without storing it
                depth_np = np.load(str(depth_path))
                if np.max(depth_np) == 0:
                    continue
            except Exception as e:
                print(f"Warning: Could not load or process {depth_path}, skipping. Error: {e}")
                continue
            valid_samples.append((rgb_path, depth_path, pose_idx))
        
        self.samples = valid_samples
        print(f"Dataset initialized with {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, pose_idx = self.samples[idx]

        rgb_np = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)

        depth_np = np.load(str(depth_path)) # Load .npy file

        # Preprocess image and depth
        img_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
        img_tensor = self.transform_func(img_tensor)

        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float() # Depth is in meters
        depth_t_resized = F.interpolate(depth_t, (256, 256), mode="nearest-exact")
        depth_t = F.interpolate(depth_t_resized, (self.feat_h, self.feat_h), mode="nearest-exact").squeeze()

        E_cv = self.cam_to_world_poses[pose_idx][:3, :]
        extrinsic_t = torch.from_numpy(E_cv).float()

        return {
            "img_tensor": img_tensor,
            "depth_t": depth_t,
            "extrinsic_t": extrinsic_t,
        }


# --------------------------------------------------------------------------- #
#  Arguments                                                                  #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Map a single environment using pre-generated data.")
parser.add_argument(
    "--dataset-dir",
    type=str,
    default="mapping/dataset",
    help="Path to the dataset directory."
)
parser.add_argument(
    "--poses-dir",
    type=str,
    default="mapping/dataset/poses",
    help="Path to the directory with .npy pose files."
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="mapping/map_output",
    help="Directory to save the trained grid and decoder."
)
parser.add_argument(
    "--epochs",
    type=int,
    default=30,
    help="Number of training epochs per environment."
)
parser.add_argument(
    "--train",
    action="store_true",
    default=False,
    help="Train the model. If False, load pre-trained model for visualization only."
)
parser.add_argument(
    "--continue",
    dest="continue_training",
    action="store_true",
    default=False,
    help="Continue training from a saved grid and decoder."
)
parser.add_argument(
    "--save",
    action="store_true",
    default=True,
    help="Save the trained voxel grids & shared decoder."
)
parser.add_argument(
    "--pca",
    action="store_true",
    help="Generate PCA visualization of voxel features."
)

parser.add_argument(
    "--vis-interval",
    type=int,
    default=1,
    help="Epoch interval for PCA visualization during training. Set to 0 to disable."
)
parser.add_argument(
    "--decoder-path",
    type=str,
    default="save_checkpoint/decoder.pt",
    help="Path to pre-trained decoder weights (implicit decoder)."
)
parser.add_argument(
    "--num-images",
    type=int,
    default=-1,
    help="Number of images to use from the dataset. -1 to use all."
)
parser.add_argument(
    "--load-run-dir",
    type=str,
    default=None,
    help="Path to a run directory to load model from for continuation or inference."
)
args = parser.parse_args()

# --------------------------------------------------------------------------- #
#  Device / DINOv3 model                                                     #
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load DINOv3 backbone and initialize model
backbone = torch.hub.load('dinov3', 'dinov3_vith16plus', source='local', weights=WEIGHT_PATH)
# backbone = torch.hub.load('dinov3', 'dinov3_vit7b16', source='local', weights=WEIGHT_PATH)
dino_model = DINOv3Wrapper(backbone, output_dim=1280).to(DEVICE).eval()

# --------------------------------------------------------------------------- #
#  Shared Decoder                                                             #
# --------------------------------------------------------------------------- #
GRID_LVLS         = 2
GRID_FEAT_DIM     = 64
LEVEL_SCALE       = 2.0
HASH_TABLE_SIZE   = 2**21
RESOLUTION        = 0.2
decoder = ImplicitDecoder(
    voxel_feature_dim=GRID_FEAT_DIM * GRID_LVLS,
    hidden_dim=240,
    output_dim=1280,
).to(DEVICE)

# ----------------------------------------------------------------------- #
#  Load pre-trained decoder weights if available                          #
# ----------------------------------------------------------------------- #
if args.decoder_path and os.path.isfile(args.decoder_path):
    try:
        state_dict = torch.load(args.decoder_path, map_location=DEVICE)['model']
        decoder.load_state_dict(state_dict)
        print(f"[INIT] Loaded pre-trained decoder weights from {args.decoder_path}")
    except Exception as e:
        print(f"[INIT] Failed to load pre-trained decoder weights from {args.decoder_path}: {e}")
else:
    print(f"[INIT] No pre-trained decoder weights found at {args.decoder_path}. Using random initialization.")

OPT_LR = 1e-3

# --------------------------------------------------------------------------- #
#  Scene bounds (should match map_table.py)                                   #
# --------------------------------------------------------------------------- #
# SCENE_MIN = (-6.5, -11.0, -0.5)
# SCENE_MAX = (7.0,  3.0,  3.0)


SCENE_MIN = (-7.0, -5.5, -1.0)
SCENE_MAX = (7.0,  9.5,  3.0)


# --------------------------------------------------------------------------- #
#  Helper functions                                                           #
# --------------------------------------------------------------------------- #
def load_w2c_poses_from_dir(dir_path: str) -> np.ndarray:
    """Loads world-to-camera poses from .npy files, returns (N, 3, 4) matrices."""
    pose_files = sorted(glob.glob(os.path.join(dir_path, "*.npy")))
    if not pose_files:
        raise FileNotFoundError(f"No .npy pose files found in {dir_path}")
    poses_4x4 = [np.load(f) for f in pose_files]
    poses_3x4 = [p[:3, :] for p in poses_4x4]  # Convert 4x4 to 3x4
    return np.stack(poses_3x4, axis=0)

# --------------------------------------------------------------------------- #
#  Main processing loop                                                       #
# --------------------------------------------------------------------------- #
def main():
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    if args.save or args.pca:
        output_path.mkdir(parents=True, exist_ok=True)

    if args.load_run_dir:
        load_path = Path(args.load_run_dir)
        print(f"[LOAD] Attempting to load model from specified run directory: {load_path}")
        assert load_path.is_dir(), f"Specified load directory does not exist: {load_path}"
    else:
        load_path = output_path

    intrinsic_path = dataset_path / "intrinsics.txt"
    if not intrinsic_path.exists():
        print(f"[ERROR] Intrinsic file not found at {intrinsic_path}.")
        return
    K = np.loadtxt(intrinsic_path)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # Adjust intrinsics for the new image size (256x256) from original (512x512)
    fx = fx * (256 / 512)
    fy = fy * (256 / 512)
    cx = cx * (256 / 512)
    cy = cy * (256 / 512)

    try:
        world_to_cam_poses = load_w2c_poses_from_dir(args.poses_dir)
    except FileNotFoundError:
        print(f"[ERROR] Poses directory '{args.poses_dir}' not found or is empty.")
        return

    # The poses are world-to-camera, but we need camera-to-world to transform points from camera to world frame.
    cam_to_world_poses_list = []
    for w2c_3x4 in world_to_cam_poses:
        c2w = np.eye(4)
        R = w2c_3x4[:3, :3]
        t = w2c_3x4[:3, 3]
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        cam_to_world_poses_list.append(c2w)
    cam_to_world_poses = np.array(cam_to_world_poses_list)


    env_name = dataset_path.name

    log_dir = output_path / f"runs/{env_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    tb_writer = SummaryWriter(log_dir=log_dir)
    print(f"[INIT] TensorBoard logging enabled. Log directory: {log_dir}")

    # 1. Create or load VoxelHashTable
    grid = None
    agg_coords = []
    
    if args.train:
        grid_path = load_path / "grid.pt"
        decoder_path = load_path / "decoder.pt"

        if args.continue_training and grid_path.exists() and decoder_path.exists():
            print(f"\n[LOAD] Found existing grid and decoder. Loading for continued training...")
            
            # Load decoder
            state_dict = torch.load(decoder_path, map_location=DEVICE)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                decoder.load_state_dict(state_dict['model'])
            else:
                decoder.load_state_dict(state_dict)
            print(f"[LOAD] Loaded decoder from {decoder_path}")

            # Load grid for continued training
            sparse_data = torch.load(grid_path, map_location=DEVICE)
            grid = VoxelHashTable(
                resolution=RESOLUTION,
                num_levels=GRID_LVLS,
                level_scale=LEVEL_SCALE,
                feature_dim=GRID_FEAT_DIM,
                hash_table_size=HASH_TABLE_SIZE,
                scene_bound_min=SCENE_MIN,
                scene_bound_max=SCENE_MAX,
                device=DEVICE,
                mode="train",
                sparse_data=sparse_data,
            )
            print(f"[LOAD] Loaded grid from {grid_path}")
        else:
            print("\n[INIT] No existing trained model found. Creating a new one.")
            # Training mode: Create new grid
            grid = VoxelHashTable(
                resolution=RESOLUTION,
                num_levels=GRID_LVLS,
                level_scale=LEVEL_SCALE,
                feature_dim=GRID_FEAT_DIM,
                hash_table_size=HASH_TABLE_SIZE,
                scene_bound_min=SCENE_MIN,
                scene_bound_max=SCENE_MAX,
                device=DEVICE,
                mode="train",
            )
            
            stats = grid.collision_stats()
            print(f"--- Collision stats for {env_name}: ---")
            for level_name, stat in stats.items():
                total = stat['total']
                collisions = stat['col']
                if total > 0:
                    percentage = (collisions / total) * 100
                    print(f"  {level_name}: {collisions} collisions out of {total} voxels ({percentage:.2f}%)")
                else:
                    print(f"  {level_name}: 0 voxels")
            print("-------------------------------------------------")
    else:
        # Inference mode: Load pre-trained grid
        print("\n[LOAD] Loading pre-trained grid for visualization...")
        sparse_grid_path = load_path / "grid.sparse.pt"
        if not sparse_grid_path.exists():
            print(f"[LOAD] [ERROR] Pre-trained grid not found at {sparse_grid_path}. Please train first.")
            return
        
        sparse_data = torch.load(sparse_grid_path, map_location=DEVICE)
        grid = VoxelHashTable(
            mode="infer",
            sparse_data=sparse_data,
            device=DEVICE,
            feature_dim=GRID_FEAT_DIM,
            hash_table_size=HASH_TABLE_SIZE
        )
        print(f"[LOAD] Loaded grid from {sparse_grid_path}")
        
        # Load decoder weights
        decoder_path = load_path / "decoder.pt"
        if not decoder_path.exists():
            print(f"[ERROR] Pre-trained decoder not found at {decoder_path}. Cannot proceed with visualization.")
            return
        
        state_dict = torch.load(decoder_path, map_location=DEVICE)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            decoder.load_state_dict(state_dict['model'])
        else:
            decoder.load_state_dict(state_dict)
        print(f"[LOAD] Loaded decoder from {decoder_path}")

    if args.train:
        # 2. Setup a single optimizer for the grid and the shared decoder
        optimizer = torch.optim.Adam(list(grid.parameters()) + list(decoder.parameters()), lr=OPT_LR)

        # 3. Load all data from the environment
        all_samples = []
        rgb_files = sorted(list((dataset_path / "rgb").glob("*.png")))
        depth_files = sorted(list((dataset_path / "depth").glob("*.npy")))
        
        if not rgb_files or not depth_files:
            print(f"[WARN] No data found in {dataset_path}, skipping.")
            return
            
        for i, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
            all_samples.append((rgb_path, depth_path, i)) # i is pose_idx

        if args.num_images > 0:
            all_samples = all_samples[:args.num_images]
            print(f"--- Using the first {args.num_images} images for training. ---")

        if not all_samples:
            print("[ERROR] No training data found. Exiting.")
            return
        
        try:
            dataset = SingleEnvDataset(all_samples, args.poses_dir, transform)
            if len(dataset) == 0:
                print("[ERROR] No valid training data found after filtering. Exiting.")
                return
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return
        cam_to_world_poses = dataset.cam_to_world_poses
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        print(f"--- Loaded {len(dataset)} samples. Starting training... ---")

        # 4. Main training loop
        LOG_INTERVAL = 100
        for epoch in range(args.epochs):
            loss_history = []
            epoch_coords = []
            
            for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
                # --- Get preprocessed data from dataset ---
                img_tensor = data["img_tensor"].to(DEVICE)
                depth_t = data["depth_t"].to(DEVICE)
                extrinsic_t = data["extrinsic_t"].unsqueeze(1).to(DEVICE)
                
                with torch.no_grad():
                    vis_feat = dino_model(img_tensor)
                
                # (NOTE): match the intrinsic of the camera,  fx fy 193.9897 double check
                coords_world, _ = get_3d_coordinates(
                    depth_t, extrinsic_t,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    original_size=256,
                )

                B, C_, Hf, Wf = vis_feat.shape
                feats_valid = vis_feat.permute(0, 2, 3, 1).reshape(-1, C_)
                coords_valid = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)
            

                in_x = (coords_valid[:,0] >= SCENE_MIN[0]) & (coords_valid[:,0] <= SCENE_MAX[0])
                in_y = (coords_valid[:,1] >= SCENE_MIN[1]) & (coords_valid[:,1] <= SCENE_MAX[1])
                in_z = (coords_valid[:,2] >= SCENE_MIN[2]) & (coords_valid[:,2] <= SCENE_MAX[2])
                in_bounds = in_x & in_y & in_z
                
                # Filter out points with depth < 0.05
                depth_flat = depth_t.reshape(-1)
                valid_depth = depth_flat >= 0.05
                in_bounds = in_bounds & valid_depth

                if in_bounds.sum() == 0:
                    continue

                coords_valid = coords_valid[in_bounds].to(DEVICE)

                # accumulate for visualization (store on CPU to save GPU mem)
                if args.pca:
                    epoch_coords.append(coords_valid.cpu())
                feats_valid = feats_valid[in_bounds].to(DEVICE)
            
                
                voxel_feat = grid.query_voxel_feature(coords_valid)
                pred_feat = decoder(voxel_feat)

                cos_sim = F.cosine_similarity(pred_feat, feats_valid, dim=-1)
                loss = 1.0 - cos_sim.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())


                global_step = epoch * len(dataloader) + i
                tb_writer.add_scalar("Loss/step", loss.item(), global_step)

                if (i + 1) % LOG_INTERVAL == 0:
                    total_steps = len(dataloader)
                    print(f"[Epoch {epoch+1}/{args.epochs}] [{i+1}/{total_steps}] Loss: {loss.item():.4f}")
            
            avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0
            print(f"[Epoch {epoch+1}/{args.epochs}] Avg. Loss: {avg_loss:.4f}")


            tb_writer.add_scalar("Loss/epoch", avg_loss, epoch + 1)

            # --- Periodic PCA Visualization ---
            if args.pca and args.vis_interval > 0 and (epoch + 1) % args.vis_interval == 0:
                run_pca_visualization(server, grid, decoder, epoch_coords, env_name, epoch + 1, DEVICE)

            # --- Save checkpoint after each epoch ---
            if args.save:
                dense_grid_path = log_dir / f"grid_epoch_{epoch+1}.pt"
                sparse_grid_path = log_dir / f"grid.sparse_epoch_{epoch+1}.pt"
                grid.save_dense(dense_grid_path)
                grid.save_sparse(sparse_grid_path)
                print(f"[SAVE] Saved grid for epoch {epoch+1} to {dense_grid_path} and {sparse_grid_path}")
                
                decoder_path = log_dir / f"decoder_epoch_{epoch+1}.pt"
                torch.save(decoder.state_dict(), decoder_path)
                print(f"[SAVE] Saved decoder for epoch {epoch+1} to {decoder_path}")

 
        # ----------------------------------------------------------------------- #
        #  Check hash collisions in infer mode                                     #
        # ----------------------------------------------------------------------- #
        print("\n[CHECK] Evaluating hash collisions in infer mode...")
        sparse_data = grid.export_sparse()
        infer_grid = VoxelHashTable(
            mode="infer", 
            sparse_data=sparse_data, 
            device=DEVICE,
            feature_dim=GRID_FEAT_DIM,
            hash_table_size=HASH_TABLE_SIZE
        )
        stats = infer_grid.collision_stats()
        print(f"  [Infer] {env_name}:")
        for level_name, stat in stats.items():
            total = stat['total']
            collisions = stat['col']
            if total > 0:
                percentage = (collisions / total) * 100
                print(f"    {level_name}: {collisions} collisions out of {total} voxels ({percentage:.2f}%)")
            else:
                print(f"    {level_name}: 0 voxels")

        if args.save:
            dense_grid_path = log_dir / "grid.pt"
            sparse_grid_path = log_dir / "grid.sparse.pt"
            grid.save_dense(dense_grid_path)
            grid.save_sparse(sparse_grid_path)
            print(f"[SAVE] Saved grid to {dense_grid_path} and {sparse_grid_path}")

            decoder_path = log_dir / "decoder.pt"
            torch.save(decoder.state_dict(), decoder_path)
            print(f"[SAVE] Saved decoder to {decoder_path}")
    else:
        # Inference mode: Load coordinates for visualization
        print("\n[VIS] Loading data for visualization...")
        rgb_files = sorted(list((dataset_path / "rgb").glob("*.png")))
        depth_files = sorted(list((dataset_path / "depth").glob("*.npy")))
        
        if args.num_images > 0:
            rgb_files = rgb_files[:args.num_images]
            depth_files = depth_files[:args.num_images]
            print(f"--- Using the first {args.num_images} images for visualization. ---")
        
        if not rgb_files or not depth_files:
            print(f"[VIS] [WARN] No data found in {env_name}, skipping.")
        else:
            img_h = 256  # New image height
            patch_size = 16  # DINOv3 patch size
            feat_h = img_h // patch_size
            for i, (rgb_path, depth_path) in enumerate(tqdm(zip(rgb_files, depth_files), desc=f"Loading {env_name}", total=len(rgb_files))):
                depth_np = np.load(str(depth_path))
                if depth_np is None or np.max(depth_np) == 0:
                    continue
                
                depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()
                depth_t = F.interpolate(depth_t, (feat_h, feat_h), mode="nearest-exact").squeeze()
                
                E_cv = cam_to_world_poses[i][:3, :]
                extrinsic_t = torch.from_numpy(E_cv).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                depth_t = depth_t.unsqueeze(0).to(DEVICE)
                
                coords_world, _ = get_3d_coordinates(
                    depth_t, extrinsic_t,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    original_size=256,
                )
                
                coords_valid = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)
                
                in_x = (coords_valid[:,0] >= SCENE_MIN[0]) & (coords_valid[:,0] <= SCENE_MAX[0])
                in_y = (coords_valid[:,1] >= SCENE_MIN[1]) & (coords_valid[:,1] <= SCENE_MAX[1])
                in_z = (coords_valid[:,2] >= SCENE_MIN[2]) & (coords_valid[:,2] <= SCENE_MAX[2])
                in_bounds = in_x & in_y & in_z
                
                # Filter out points with depth < 0.05
                depth_flat = depth_t.reshape(-1)
                valid_depth = depth_flat >= 0.05
                in_bounds = in_bounds & valid_depth
                
                if in_bounds.sum() == 0:
                    continue
                
                coords_valid = coords_valid[in_bounds]
                agg_coords.append(coords_valid.cpu())
        
        print(f"[VIS] Loaded coordinates for visualization.")
    
    # --- PCA Visualization ---
    if args.pca:
        intrinsic_path = dataset_path / "intrinsics.txt"
        if not intrinsic_path.exists():
            print(f"[VIS] [ERROR] Intrinsic file not found at {intrinsic_path}. Skipping PCA.")
        else:
            if args.train:
                # In training mode, periodic visualization is used. A final visualization can be triggered if needed.
                if args.vis_interval == 0:
                     print("\n[VIS] To see PCA visualization during training, set a value for --vis-interval (e.g., --vis-interval 5).")
            else:
                # In inference mode, run PCA on all aggregated coordinates.
                run_pca_visualization(server, grid, decoder, agg_coords, env_name, "final", DEVICE)

    tb_writer.close()

    if args.pca:
        print("\n[VIS] Viser server is running. Open the link in your browser.")
        while True:
            time.sleep(1)


    print("\nDone.")

if __name__ == "__main__":
    main()
