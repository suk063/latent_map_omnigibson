import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import math


def depth_to_point_cloud(depth_image_np, rgb_image_np, K):
    depth = torch.as_tensor(depth_image_np, dtype=torch.float32)
    rgb = torch.as_tensor(
        rgb_image_np.astype(np.float32)/255.0 if rgb_image_np.dtype==np.uint8 else rgb_image_np,
        dtype=torch.float32
    )
    device = depth.device  # 보통 cpu; 필요 시 caller가 .to(device) 하세요

    H, W = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    z = depth.reshape(-1)
    valid = torch.isfinite(z) & (z > 0)
    xs, ys, z = xs.reshape(-1)[valid], ys.reshape(-1)[valid], z[valid]
    colors = rgb.reshape(-1,3)[valid]

    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
    fx, fy = Kt[0,0], Kt[1,1]
    cx, cy = Kt[0,2], Kt[1,2]

    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    pts = torch.stack([X, Y, z], dim=1)   # OpenCV cam frame (x right, y down, z forward)
    return pts, colors

def transform_point_cloud(points, pose, orientation, normalize=True):
    pts = torch.as_tensor(points, dtype=torch.float32)
    device, dtype = pts.device, pts.dtype

    t = torch.as_tensor(pose, dtype=dtype, device=device).view(3)
    q = torch.as_tensor(orientation, dtype=dtype, device=device).view(4)

    if normalize:
        q = q / (q.norm() + 1e-12)

    x, y, z, w = q
    Rm = torch.stack([
        torch.stack([1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)]),
        torch.stack([2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)]),
        torch.stack([2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)])
    ])  # (3,3)

    rotated = pts @ Rm.T
    return rotated + t