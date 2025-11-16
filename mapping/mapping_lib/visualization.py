import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import plotly.graph_objects as go
import os


def run_pca_visualization(server, grid, decoder, coords_list, env_name, epoch_label, device, config):
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

    print(f"[VIS] Updating PCA visualization for {env_name} (epoch: {epoch_label}) to Viser.")
    server.add_point_cloud(
        name=f"/pca/{env_name}",
        points=centered_vertices,
        colors=(filtered_colors * 255).astype(np.uint8),
        point_size=0.01
    )


def visualize_instances_plotly(grid, coords_list, env_name, epoch_label, device, config, target_ids_to_vis):
    """Visualizes instance IDs using Plotly."""
    if not coords_list:
        print("[VIS-Plotly] No coordinates to visualize.")
        return

    print(f"\n[VIS-Plotly] Running instance visualization for {env_name} (epoch: {epoch_label})...")

    vertices_vis = torch.cat(coords_list, dim=0).numpy()

    max_points_for_vis = 2000000
    if vertices_vis.shape[0] > max_points_for_vis:
        print(f"[VIS-Plotly] Downsampling from {vertices_vis.shape[0]} to {max_points_for_vis} points.")
        indices = np.random.choice(vertices_vis.shape[0], max_points_for_vis, replace=False)
        vertices_vis = vertices_vis[indices]

    if vertices_vis.shape[0] == 0:
        print("[VIS-Plotly] No vertices to visualize.")
        return

    coords_t = torch.from_numpy(vertices_vis).to(device)
    batch_size = 50000

    # Query instance IDs
    all_inst_ids = []
    with torch.no_grad():
        for i in tqdm(range(0, coords_t.shape[0], batch_size), desc="[VIS-Plotly] Querying instance IDs"):
            batch_coords = coords_t[i:i + batch_size]
            inst_ids_corners = grid.query_instance_ids(batch_coords)
            all_inst_ids.append(inst_ids_corners)

    if not all_inst_ids:
        print("[VIS-Plotly] No instance IDs queried.")
        return

    inst_ids_corners = torch.cat(all_inst_ids, dim=0)
    inst_ids_mode, _ = torch.mode(inst_ids_corners, dim=1)

    # Filter for target instances
    target_ids = torch.tensor(target_ids_to_vis, device=device, dtype=torch.long)
    mask = torch.isin(inst_ids_mode, target_ids)

    instance_points = vertices_vis[mask.cpu().numpy()]
    instance_ids = inst_ids_mode[mask].cpu().numpy()

    if instance_points.shape[0] == 0:
        print("[VIS-Plotly] No points found for target instances.")
        return

    # Create color map
    color_map = {
        28: 'red', 29: 'green', 30: 'blue', 31: 'yellow',
        32: 'cyan', 33: 'magenta', 34: 'orange',
    }
    instance_colors = [color_map.get(id, 'gray') for id in instance_ids]

    # Create Plotly figure
    fig = go.Figure(data=[go.Scatter3d(
        x=instance_points[:, 0], y=instance_points[:, 1], z=instance_points[:, 2],
        mode='markers',
        marker=dict(size=2, color=instance_colors, opacity=0.8)
    )])

    fig.update_layout(
        title=f"Instance Visualization for {env_name} (Epoch {epoch_label})",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )

    # Save to HTML
    output_dir = config.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"instance_vis_{env_name}_epoch_{epoch_label}.html")
    fig.write_html(filepath)
    print(f"[VIS-Plotly] Saved instance visualization to {filepath}")
