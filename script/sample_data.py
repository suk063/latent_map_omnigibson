#!/usr/bin/env python3
import os
import gc
import sys
import argparse
import random
import numpy as np
import open3d as o3d


# ---- Recommended environment variables (to mitigate contention for render/replicator/unload) ----
os.environ.setdefault("OMNI_KIT_HEADLESS", "1")          # headless
os.environ.setdefault("CARB_LOG", "warn")                # Suppress logs
# Optionally, disable replicator-related plugins (works without them)
os.environ.setdefault("OMNI_KIT_DISABLE_PLUGINS", "omni.syntheticdata.plugin,omni.replicator.core,omni.replicator.isaac")

import omnigibson as og
from omnigibson import lazy
from omnigibson.maps.traversable_map import TraversableMap
from omnigibson.utils.asset_utils import get_available_behavior_1k_scenes
from omnigibson.utils.ui_utils import choose_from_options


def get_traversable_xy_batch(trav_map: TraversableMap, n: int):
    """
    Samples n (x,y) coordinates from the TraversableMap.
    """
    points_xy = []
    for _ in range(n):
        # get_random_point returns (floor_idx, [x,y,z])
        _, pos_3d = trav_map.get_random_point(floor=0)
        points_xy.append(pos_3d[:2])
    return np.array(points_xy, dtype=np.float32)


def get_random_xy_batch(env, n: int):
    """
    Samples n (x,y) coordinates from the scene's AABB.
    """
    low, high = lazy.omni.usd.get_context().compute_path_world_bounding_box(env.scene.prim_path)
    x_min, y_min = float(low[0]), float(low[1])
    x_max, y_max = float(high[0]), float(high[1])

    x_coords = np.random.uniform(x_min, x_max, size=(n,))
    y_coords = np.random.uniform(y_min, y_max, size=(n,))
    
    return np.stack([x_coords, y_coords], axis=1).astype(np.float32)


def sample_free_space_points_batch(env,
                                   trav_map: TraversableMap,
                                   desired_n: int = 2000,
                                   batch_xy: int = 2000,
                                   radius: float = 0.01,
                                   xy_mode: str = "traversable"):
    """
    Samples a large number of free space points.
    - desired_n: The final number of points to obtain
    - batch_xy: Number of (x,y) candidates to generate at once
    - radius: Radius of the sphere for collision checking
    - xy_mode: "traversable" (from traversable map) or "aabb" (from scene AABB)
    """
    low, high = lazy.omni.usd.get_context().compute_path_world_bounding_box(env.scene.prim_path)
    z_min, z_max = float(low[2]), float(high[2])

    results = []
    # Warm-up
    for _ in range(3):
        og.sim.step()

    while len(results) < desired_n:
        # 1) Sample a batch of (x, y)
        if xy_mode == "traversable":
            xy = get_traversable_xy_batch(trav_map, batch_xy)  # [B,2]
        elif xy_mode == "aabb":
            xy = get_random_xy_batch(env, batch_xy)
        else:
            raise ValueError(f"Unknown xy_mode: {xy_mode}")

        # 2) z batch
        z = np.random.uniform(z_min, z_max, size=(xy.shape[0],)).astype(np.float32)

        pts = np.concatenate([xy, z[:, None]], axis=1)  # [B,3]

        # 3) Collision check
        for i in range(pts.shape[0]):
            p = pts[i]
            if not og.sim.psqi.overlap_sphere_any(radius=radius, pos=p):
                results.append(p.copy())
                if len(results) >= desired_n:
                    break

        print(f"[batch] gathered {len(results)}/{desired_n}")
        og.sim.step()

    return np.asarray(results, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100000, help="Number of free-space points to sample")
    parser.add_argument("--scene", type=str, help="Scene model name (e.g., Rs_int). If not specified, it will be selected interactively.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (if reproducibility is needed)")
    parser.add_argument("--output", type=str, default=None, help="Output .ply file path. If not specified, it is automatically generated based on the scene name.")
    parser.add_argument("--batch-xy", type=int, default=2000, help="Number of (x,y) candidates to generate at once")
    parser.add_argument("--radius", type=float, default=0.01, help="Collision check sphere radius")
    parser.add_argument("--xy-mode", type=str, default="aabb", choices=["traversable", "aabb"], help="xy sampling method")
    args = parser.parse_args()

    # RNG
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Scene selection
    available_scenes = get_available_behavior_1k_scenes()
    if args.scene:
        if args.scene not in available_scenes:
            print(f"Error: Scene '{args.scene}' not found. Available scenes: {list(available_scenes)}")
            return
        scene_model = args.scene
    else:
        scene_model = choose_from_options(options=available_scenes, name="scene")

    config = {
        "scene": {"type": "InteractiveTraversableScene", "scene_model": scene_model},
        "robots": [],
        "headless": True,
    }

    env = None
    env = og.Environment(configs=config)
    env.reset()
    for _ in range(5):
        og.sim.step()

    trav_map = env.scene.trav_map
    if trav_map is None:
        print("Building traversable map...")
        env.scene.build_traversable_map()
        trav_map = env.scene.trav_map
        print("Traversable map built.")

    points = []
    print(f"Sampling {args.num} points...")
    points = sample_free_space_points_batch(
        env=env,
        trav_map=trav_map,
        desired_n=args.num,
        batch_xy=args.batch_xy,
        radius=args.radius,
        xy_mode=args.xy_mode,
    )

    og.sim.stop()

    print("Sampling finished.")
    points_arr = np.array(points, dtype=np.float32)

    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_arr)
    
    # Save to file
    if args.output:
        output_path = args.output
    else:
        output_path = f"{scene_model}_points.ply"
    
    print(f"Saving points to {output_path}...")
    o3d.io.write_point_cloud(output_path, pcd)
    print("Saved.")
    
    print("Visualizing points with Open3D...")
    o3d.visualization.draw_geometries([pcd])
    print("Visualization finished.")


if __name__ == "__main__":
    main()
