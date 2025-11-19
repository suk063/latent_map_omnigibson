import os, time, argparse, glob
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import viser
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm   
from torch.utils.tensorboard import SummaryWriter
import open_clip
import yaml
import json

# local modules
from mapping_lib.utils import get_3d_coordinates
from mapping_lib.voxel_hash_table import VoxelHashTable
from mapping_lib.implicit_decoder import ImplicitDecoder
from mapping_lib.vision_wrapper import EvaClipWrapper, DINOv3Wrapper
from mapping_lib.visualization import run_pca_visualization, visualize_instances_plotly
from torchvision import transforms


# ==============================================================================================
#  Dataset Class                                                              #
# ==============================================================================================
class MultiEnvDataset(Dataset):
    def __init__(self, dataset_dir, target_envs, num_images, transform_func, image_size, patch_size):
        self.transform_func = transform_func
        self.image_size = image_size
        self.patch_size = patch_size
        self.feat_h = self.image_size // patch_size

        self.samples = []
        self.cam_to_world_poses = {}
        self.env_names = []

        if not target_envs:
            env_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        else:
            env_dirs = sorted([dataset_dir / env for env in target_envs])

        print(f"\nFound {len(env_dirs)} environments.")

        for env_dir in env_dirs:
            env_name = env_dir.name
            
            print(f"--- Processing environment: {env_name} ---")
            
            poses_dir = env_dir / "poses"
            rgb_dir = env_dir / "rgb"
            depth_dir = env_dir / "depth"
            inst_id_dir = env_dir / "seg_instance_id"

            if not all([d.exists() for d in [poses_dir, rgb_dir, depth_dir, inst_id_dir]]):
                print(f"  [WARN] Missing poses, rgb, depth, or seg_instance_id directory in {env_dir}. Skipping.")
                continue
            
            self.env_names.append(env_name)

            # Load and process poses
            pose_files = sorted(list(poses_dir.glob("*.npy")))
            if not pose_files:
                print(f"  [WARN] No .npy pose files found in '{poses_dir}'. Skipping env.")
                self.env_names.pop()
                continue
            
            poses_4x4 = [np.load(f) for f in pose_files]
            cam_to_world_poses_list = [np.linalg.inv(p_4x4) for p_4x4 in poses_4x4]
            self.cam_to_world_poses[env_name] = np.array(cam_to_world_poses_list)

            # Collect and validate samples
            rgb_files = sorted(list(rgb_dir.glob("*.png")))
            if num_images > 0:
                print(f"  -> Using the first {num_images} images.")
                rgb_files = rgb_files[:num_images]

            env_samples = []
            for i, rgb_path in enumerate(tqdm(rgb_files, desc=f"  Validating {env_name}")):
                depth_path = depth_dir / f"{rgb_path.stem}.npy"
                inst_id_path = inst_id_dir / f"{rgb_path.stem}.npy"
                if not depth_path.exists() or not inst_id_path.exists():
                    continue
                env_samples.append({"rgb_path": rgb_path, "depth_path": depth_path, "inst_id_path": inst_id_path, "pose_idx": i, "env_name": env_name})
            
            self.samples.extend(env_samples)
        
        print(f"\nDataset initialized with {len(self.samples)} total samples from {len(self.env_names)} environments.")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        rgb_path = sample_info["rgb_path"]
        depth_path = sample_info["depth_path"]
        inst_id_path = sample_info["inst_id_path"]
        pose_idx = sample_info["pose_idx"]
        env_name = sample_info["env_name"]

        try:
            rgb_np = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb_np is None: raise IOError(f"Failed to read image: {rgb_path}")
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
            
            depth_np = np.load(str(depth_path))
            inst_id_np = np.load(str(inst_id_path))
            if np.max(depth_np) == 0:
                return None # This will be filtered by the collate function

        except Exception as e:
            print(f"Warning: Error loading data for sample {idx} ({rgb_path}), skipping. Error: {e}")
            return None

        img_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
        img_tensor = self.transform_func(img_tensor)

        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()
        depth_t_resized = F.interpolate(depth_t, (self.image_size, self.image_size), mode="nearest-exact")
        depth_t = F.interpolate(depth_t_resized, (self.feat_h, self.feat_h), mode="nearest-exact").squeeze()

        inst_id_t = torch.from_numpy(inst_id_np).unsqueeze(0).unsqueeze(0).long()
        inst_id_t_resized = F.interpolate(inst_id_t.float(), (self.image_size, self.image_size), mode="nearest-exact").long()
        inst_id_t = F.interpolate(inst_id_t_resized.float(), (self.feat_h, self.feat_h), mode="nearest-exact").long().squeeze()

        E_cv = self.cam_to_world_poses[env_name][pose_idx][:3, :]
        extrinsic_t = torch.from_numpy(E_cv).float()

        return {
            "img_tensor": img_tensor,
            "depth_t": depth_t,
            "inst_id_t": inst_id_t,
            "extrinsic_t": extrinsic_t,
            "env_name": env_name,
        }

def collate_fn(batch):
    """
    Custom collate function to filter out None values from the batch.
    This is used to handle cases where `__getitem__` returns None for invalid samples.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)


# --------------------------------------------------------------------------- #
#  Arguments                                                                  #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Map a single environment using a configuration file.")
parser.add_argument(
    "--config",
    type=str,
    default="mapping/config.yaml",
    help="Path to the YAML configuration file."
)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# --------------------------------------------------------------------------- #
#  Device                                                                     #
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
#  Scene bounds (loaded from config)                                          #
# --------------------------------------------------------------------------- #
SCENE_MIN = tuple(config['scene_min'])
SCENE_MAX = tuple(config['scene_max'])


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

    # --------------------------------------------------------------------------- #
    #  Model-specific configurations                                              #
    # --------------------------------------------------------------------------- #
    if config['model_type'] == "clip":
        model_config = config['clip_model']
        image_size = model_config['image_size']
        patch_size = model_config['patch_size']
        feature_dim = model_config['feature_dim']
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ])
        
        # Load EVA_CLIP backbone and initialize model
        clip_model_name  = model_config['name']
        clip_weights_id  = model_config['weights_id']
        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_weights_id
        )
        clip_model = clip_model.to(DEVICE).eval()
        model = EvaClipWrapper(clip_model, output_dim=feature_dim).to(DEVICE).eval()
        print("[INIT] Loaded EVA-CLIP model.")

    elif config['model_type'] == "dino":
        model_config = config['dino_model']
        image_size = model_config['image_size']
        patch_size = model_config['patch_size']
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ])

        # Load DINOv3 backbone and initialize model
        WEIGHT_PATH = model_config['weights_path']
        backbone = torch.hub.load('dinov3', 'dinov3_vith16plus', source='local', weights=WEIGHT_PATH)
        model = DINOv3Wrapper(backbone).to(DEVICE).eval()
        feature_dim = model.feature_dim
        print(f"[INIT] Loaded DINOv3 model with feature dimension {feature_dim}.")
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    dataset_path = Path(config['dataset_dir'])
    output_path = Path(config['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    load_path = output_path

    # Determine target environments
    target_envs = config.get('target_envs', [])
    if not target_envs:
        env_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        env_names = [d.name for d in env_dirs]
        print(f"[INIT] No target_envs specified. Using all {len(env_names)} environments found in {dataset_path}")
    else:
        env_names = sorted(target_envs)
        print(f"[INIT] Using specified target_envs: {env_names}")

    if not env_names:
        print("[ERROR] No environments found or specified to process. Exiting.")
        return
    
    first_env_path = dataset_path / env_names[0]
    intrinsic_path = first_env_path / "intrinsics.txt"
    if not intrinsic_path.exists():
        print(f"[ERROR] Intrinsic file not found at {intrinsic_path} for the first environment. Exiting.")
        return
    K = np.loadtxt(intrinsic_path)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # Adjust intrinsics for the new image size from original
    original_size = config['dataset']['original_image_size']
    fx = fx * (image_size / original_size)
    fy = fy * (image_size / original_size)
    cx = cx * (image_size / original_size)
    cy = cy * (image_size / original_size)


    # Pose loading is now handled by the dataset class
    
    task_name = dataset_path.name
    log_dir = output_path / f"runs/{task_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    tb_writer = SummaryWriter(log_dir=log_dir)
    print(f"[INIT] TensorBoard logging enabled. Log directory: {log_dir}")

    # --------------------------------------------------------------------------- #
    #  Shared Decoder                                                             #
    # --------------------------------------------------------------------------- #
    grid_config = config['grid']
    decoder_config = config['decoder']
    GRID_LVLS         = grid_config['levels']
    GRID_FEAT_DIM     = grid_config['feature_dim']
    LEVEL_SCALE       = grid_config['level_scale']
    HASH_TABLE_SIZE   = grid_config['hash_table_size']
    RESOLUTION        = grid_config['resolution']
    decoder = ImplicitDecoder(
        voxel_feature_dim=GRID_FEAT_DIM * GRID_LVLS,
        hidden_dim=decoder_config['hidden_dim'],
        output_dim=feature_dim,
    ).to(DEVICE)

    OPT_LR = config['training']['optimizer_lr']
    
    # 1. Create VoxelHashTables for each environment
    grids = {}
    for env_name in env_names:
        print(f"\n[INIT] Creating a new grid for {env_name}.")
        grid = VoxelHashTable(
            resolution=RESOLUTION, num_levels=GRID_LVLS, level_scale=LEVEL_SCALE,
            feature_dim=GRID_FEAT_DIM, hash_table_size=HASH_TABLE_SIZE,
            scene_bound_min=SCENE_MIN, scene_bound_max=SCENE_MAX,
            device=DEVICE,
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
        grids[env_name] = grid

    # 2. Setup a single optimizer for all grids and the shared decoder
    all_params = list(decoder.parameters())
    for grid in grids.values():
        all_params.extend(list(grid.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=OPT_LR)

    # 3. Load all data from the environments
    try:
        dataset = MultiEnvDataset(dataset_path, target_envs, config['num_images'], transform, image_size, patch_size)
        if len(dataset) == 0:
            print("[ERROR] No valid training data found after filtering. Exiting.")
            return
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print(f"--- Loaded {len(dataset)} samples. Starting training... ---")

    # --- Setup target instance IDs for storing ---
    target_instance_ids = list(range(28, 35))  # IDs 28 to 34
    print(f"[INFO] Target instance IDs to store: {target_instance_ids}")
    target_instance_ids_tensor = torch.tensor(target_instance_ids, device=DEVICE, dtype=torch.long)


    # ==============================================================================================
    #  Training Loop                                                              #
    # ==============================================================================================
    LOG_INTERVAL = config['training']['log_interval']
    for epoch in range(config['training']['epochs']):
        loss_history = []
        epoch_coords = defaultdict(list)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for i, data in enumerate(pbar):
            if not data: continue # Skip empty batches from collate_fn

            # --- Get preprocessed data from dataset ---
            img_tensor = data["img_tensor"].to(DEVICE)
            depth_t = data["depth_t"].to(DEVICE)
            extrinsic_t = data["extrinsic_t"].to(DEVICE)
            inst_id_t = data["inst_id_t"].to(DEVICE)
            env_name = data["env_name"][0] # Batch size is 1

            with torch.no_grad():
                vis_feat = model(img_tensor)
            
            coords_world, _ = get_3d_coordinates(
                depth_t, extrinsic_t,
                fx=fx, fy=fy, cx=cx, cy=cy,
                original_size=image_size,
            )

            B, C_, Hf, Wf = vis_feat.shape
            feats_valid = vis_feat.permute(0, 2, 3, 1).reshape(-1, C_)
            coords_valid = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)
        

            in_x = (coords_valid[:,0] > SCENE_MIN[0]) & (coords_valid[:,0] < SCENE_MAX[0])
            in_y = (coords_valid[:,1] > SCENE_MIN[1]) & (coords_valid[:,1] < SCENE_MAX[1])
            in_z = (coords_valid[:,2] > SCENE_MIN[2]) & (coords_valid[:,2] < SCENE_MAX[2])
            in_bounds = in_x & in_y & in_z
            
            # Filter out points with depth < 0.01
            depth_flat = depth_t.reshape(-1)
            valid_depth = depth_flat >= 0.01
            in_bounds = in_bounds & valid_depth

            if in_bounds.sum() == 0:
                continue

            coords_valid = coords_valid[in_bounds].to(DEVICE)
            inst_id_flat = inst_id_t.reshape(-1)
            inst_id_valid = inst_id_flat[in_bounds]

            # accumulate for visualization (store on CPU to save GPU mem)
            if config['run_pca']:
                epoch_coords[env_name].append(coords_valid.cpu())
            feats_valid = feats_valid[in_bounds].to(DEVICE)
        
            
            grid = grids[env_name] # Get the correct grid for this environment

            # --- Update instance IDs in the grid for target objects ---
            if target_instance_ids_tensor.numel() > 0:
                is_target_instance = torch.isin(inst_id_valid, target_instance_ids_tensor)
                coords_to_store = coords_valid[is_target_instance]
                inst_ids_to_store = inst_id_valid[is_target_instance]

                if coords_to_store.shape[0] > 0:
                    unique_ids, counts = torch.unique(inst_ids_to_store, return_counts=True)
                    # print(f"[INFO] Found {coords_to_store.shape[0]} points for target instances. IDs: {unique_ids.cpu().numpy()}, Counts: {counts.cpu().numpy()}")
                    grid.update_instance_ids(coords_to_store, inst_ids_to_store)

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
                pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
        
        avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0
        print(f"[Epoch {epoch+1}/{config['training']['epochs']}] Avg. Loss: {avg_loss:.4f}")


        tb_writer.add_scalar("Loss/epoch", avg_loss, epoch + 1)

        # --- Save point cloud for each environment ---
        if epoch_coords:
            for env_name, coords_list in epoch_coords.items():
                if coords_list:
                    all_coords = torch.cat(coords_list, dim=0).numpy()
                    env_log_dir = log_dir / env_name
                    env_log_dir.mkdir(parents=True, exist_ok=True)
                    save_path = env_log_dir / "point_cloud.npy"
                    np.save(save_path, all_coords)
                    print(f"[SAVE] Saved point cloud for {env_name} epoch {epoch+1} to {save_path}")

        # --- Periodic PCA Visualization ---
        if config['run_pca'] and config['vis_interval'] > 0 and (epoch + 1) % config['vis_interval'] == 0:
            first_env_name = env_names[0]
            run_pca_visualization(server, grids[first_env_name], decoder, epoch_coords[first_env_name], first_env_name, epoch + 1, DEVICE, config)
            visualize_instances_plotly(grids[first_env_name], epoch_coords[first_env_name], first_env_name, epoch + 1, DEVICE, config, target_instance_ids)

        # --- Save checkpoint after each epoch ---
        for env_name, grid in grids.items():
            env_log_dir = log_dir / env_name
            env_log_dir.mkdir(parents=True, exist_ok=True)
            
            grid_path = env_log_dir / "grid.pt"
            torch.save(grid.state_dict(), grid_path)
            print(f"[SAVE] Saved latest grid for {env_name} epoch {epoch+1} to {grid_path}")

        decoder_path = log_dir / "decoder.pt"
        torch.save(decoder.state_dict(), decoder_path)
        print(f"[SAVE] Saved latest decoder for epoch {epoch+1} to {decoder_path}")


    # ----------------------------------------------------------------------- #
    #  Final save and hash collision check                                    #
    # ----------------------------------------------------------------------- #
    for env_name, grid in grids.items():
        print(f"\n[CHECK] Evaluating hash collisions for {env_name}...")
        stats = grid.collision_stats()
        print(f"  [Stats] {env_name}:")
        for level_name, stat in stats.items():
            total, collisions = stat['total'], stat['col']
            if total > 0:
                percentage = (collisions / total) * 100
                print(f"    {level_name}: {collisions} collisions out of {total} voxels ({percentage:.2f}%)")
            else:
                print(f"    {level_name}: 0 voxels")

    tb_writer.close()

    if config['run_pca']:
        print("\n[VIS] Viser server is running. Open the link in your browser.")
        while True:
            time.sleep(1)


    print("\nDone.")

if __name__ == "__main__":
    main()
