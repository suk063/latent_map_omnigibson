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
    x_min, x_max = -25.0, 28.0
    y_min, y_max = -3.0, 43.0

    x_coords = np.random.uniform(x_min, x_max, size=(n,))
    y_coords = np.random.uniform(y_min, y_max, size=(n,))
    
    return np.stack([x_coords, y_coords], axis=1).astype(np.float32)


def sample_z_mixed(n, beta_upper=0.5):
    """
    Samples z coordinates from a mixed distribution.
    - 80% uniform from [0m, 2.8m]
    - 20% exponential from [2.8m, 20m] peaked at 2.8m
    """
    n_lower = int(n * 0.8)
    n_upper = n - n_lower

    # 1. Lower uniform part
    z_lower = np.random.uniform(0.0, 2.8, size=n_lower)

    # 2. Upper exponential part
    z_upper = []
    while len(z_upper) < n_upper:
        # Oversample to reduce the number of loops
        oversample_factor = 1.5
        num_to_sample = int((n_upper - len(z_upper)) * oversample_factor)
        if num_to_sample == 0:
            num_to_sample = 1
        
        x = np.random.exponential(scale=beta_upper, size=num_to_sample)
        z = 2.8 + x
        valid_z = z[(z >= 2.8) & (z <= 20.0)]
        z_upper.extend(valid_z.tolist())
    if z_upper and len(z_upper) > n_upper:
        z_upper = z_upper[:n_upper]

    # Combine and shuffle
    z_samples = np.concatenate([z_lower, np.array(z_upper)])
    np.random.shuffle(z_samples)
    
    return z_samples.astype(np.float32)


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
        z = sample_z_mixed(n=xy.shape[0])

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
    parser.add_argument("--scene", type=str, default="house_single_floor", help="Scene model name (e.g., Rs_int). If not specified, it will be selected interactively.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (if reproducibility is needed)")
    parser.add_argument("--output", type=str, default=None, help="Output .ply file path. If not specified, it is automatically generated based on the scene name.")
    parser.add_argument("--batch-xy", type=int, default=2000, help="Number of (x,y) candidates to generate at once")
    parser.add_argument("--radius", type=float, default=0.1, help="Collision check sphere radius")
    parser.add_argument("--xy-mode", type=str, default="aabb", choices=["traversable", "aabb"], help="xy sampling method")
    
    parser.add_argument("--activity_name", type=str, default="sorting_household_items", help="BEHAVIOR activity to load. If specified, objects for the task will be loaded.")
    parser.add_argument("--activity_definition_id", type=int, default=0, help="Definition ID for the activity.")
    parser.add_argument("--activity_instance_id", type=int, default=0, help="Instance ID for the activity.")
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
        "robots": [{
            "type": "Fetch",
        }],
        "headless": True,
    }
    
    # If an activity is specified, load task-relevant objects
    if args.activity_name:
        config["scene"]["load_task_relevant_only"] = False
        # config["scene"]["not_load_object_categories"] = ["ceilings"]
        config["task"] = {
            "type": "BehaviorTask",
            "activity_name": args.activity_name,
            "activity_definition_id": args.activity_definition_id,
            "activity_instance_id": args.activity_instance_id,
            "online_object_sampling": False,
        }

    env = og.Environment(configs=config)
    robot = env.robots[0]
    robot.set_position_orientation(position=np.array([-1.0, -10.0, 2.0]))
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
