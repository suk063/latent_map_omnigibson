import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
#  small helpers                                                              #
# --------------------------------------------------------------------------- #
def _primes(dev):  # 3-tuple of large primes
    return torch.tensor([73856093, 19349669, 83492791], device=dev, dtype=torch.long)


def _corner_offsets(dev):  # (8,3) corner offsets
    return torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        device=dev,
        dtype=torch.long,
    )


# --------------------------------------------------------------------------- #
#  dense level (train)                                                        #
# --------------------------------------------------------------------------- #
class _TrainLevel(nn.Module):
    def __init__(self, res, d, buckets, smin, smax, primes, dev):
        super().__init__()
        self.res, self.d, self.buckets = res, d, buckets

        self.register_buffer("smin", torch.tensor(smin).float().to(dev), persistent=False)
        self.smin: torch.Tensor
        self.register_buffer("smax", torch.tensor(smax).float().to(dev), persistent=False)
        self.smax: torch.Tensor

        self.register_buffer("primes", primes, persistent=False)
        self.primes: torch.Tensor

        xs = torch.arange(smin[0], smax[0], res, device=dev)
        ys = torch.arange(smin[1], smax[1], res, device=dev)
        zs = torch.arange(smin[2], smax[2], res, device=dev)
        self.register_buffer(
            "grid_shape",
            torch.tensor((xs.numel(), ys.numel(), zs.numel()), device=dev),
            persistent=False,
        )
        self.grid_shape: torch.Tensor

        # indexing="ij" for (x,y,z) meshgrid
        # [(x,y,z) for x in xs for y in ys for z in zs]
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")

        self.register_buffer("coords", torch.stack([gx, gy, gz], -1).view(-1, 3), persistent=False)
        self.coords: torch.Tensor
        self.N = self.coords.size(0)

        self.register_buffer("corner_offsets", _corner_offsets(dev), persistent=False)
        self.corner_offsets: torch.Tensor

        # Hash mapping: coords to buckets
        logging.info("Using hash mapping for voxel features.")
        idx = torch.floor((self.coords - self.smin) / self.res).long()
        hv = (idx * self.primes).sum(-1) % self.buckets
        # empty = self.hash2vox[hv] == -1
        # self.hash2vox[hv[empty]] = torch.arange(self.N, device=self.voxel_features.device)[empty]
        dup = hv.unique(return_counts=True)[1] > 1
        n_collisions = int(dup.sum())

        self.voxel_features = nn.Parameter(torch.zeros(self.buckets, self.d, device=dev).normal_(0, 0.01))

        self.register_buffer("col", torch.tensor(n_collisions, device=self.voxel_features.device), persistent=False)
        self.col: torch.Tensor

        self.register_buffer("access", torch.zeros(self.buckets, dtype=torch.bool, device=dev), persistent=True)
        self.access: torch.BoolTensor

        logging.info(f"Level filled: {self.buckets} voxels, {n_collisions} collisions")

    # ---------- public utils
    @torch.no_grad()  # short stats
    def collision_stats(self):
        return dict(total=self.N, col=int(self.col))

    @torch.no_grad()
    def get_accessed_indices(self):
        return torch.nonzero(self.access).flatten()

    @torch.no_grad()  # clear log
    def reset_access_log(self):
        self.access.zero_()

    # ---------- internals
    def _lookup(self, idxg, mark_accessed: bool = True):
        vid = (idxg * self.primes).sum(-1) % self.buckets
        if mark_accessed:
            self.access[vid] = True  # log access
        return self.voxel_features[vid]

        # hv = (idxg * self.primes).sum(-1) % self.buckets
        # vid = self.hash2vox[hv]
        # valid = vid >= 0
        # out = torch.zeros(*idxg.shape[:-1], self.d, device=self.voxel_features.device, dtype=self.voxel_features.dtype)
        # if valid.any():
        #     self.access[vid[valid]] = True
        #     out[valid] = self.voxel_features[vid[valid]]
        # return out

    def query(self, pts, mark_accessed: bool = True):
        with torch.no_grad():
            q_detached = (pts - self.smin) / self.res
            base = torch.floor(q_detached).long()
            idx = base[:, None, :] + self.corner_offsets[None, :, :]

        feat = self._lookup(idx, mark_accessed=mark_accessed)

        # Recompute q with gradients if needed
        q = (pts - self.smin) / self.res
        frac = q - base.float()
        wx = torch.stack([1 - frac[:, 0], frac[:, 0]], 1)
        wy = torch.stack([1 - frac[:, 1], frac[:, 1]], 1)
        wz = torch.stack([1 - frac[:, 2], frac[:, 2]], 1)
        w = wx[:, [0, 1, 0, 1, 0, 1, 0, 1]] * wy[:, [0, 0, 1, 1, 0, 0, 1, 1]] * wz[:, [0, 0, 0, 0, 1, 1, 1, 1]]
        return (feat * w.unsqueeze(-1)).sum(1)


# --------------------------------------------------------------------------- #
#  public pyramid                                                             #
# --------------------------------------------------------------------------- #
class VoxelHashTable(nn.Module):
    """
    mode='train' → dense levels, mode='infer' → sparse levels
    """

    def __init__(
        self,
        resolution: float = 0.12,
        num_levels: int = 2,
        level_scale: float = 2.0,
        feature_dim: int = 64,
        hash_table_size: int = 2**21,
        scene_bound_min: tuple[float, ...] = (-0.8, -1.0, -0.1),
        scene_bound_max: tuple[float, ...] = (0.4,  1.0,  0.3),
        device: str = "cuda:0",
    ):
        super().__init__()
        self.d = feature_dim
        dev = torch.device(device)
        primes = _primes(dev)
        self.levels = nn.ModuleList()


        self.scene_bound_min = scene_bound_min
        self.scene_bound_max = scene_bound_max

        # Always create the level structures first.
        for lv in range(num_levels):
            res = resolution * (level_scale ** (num_levels - 1 - lv))
            self.levels.append(
                _TrainLevel(
                    res, feature_dim, hash_table_size, scene_bound_min, scene_bound_max, primes, dev
                )
            )

    # forward -----------------------------------------------------------------
    def query_voxel_feature(self, pts, mark_accessed: bool = True):  # (M,3) → (M, d*L)
        per = [lv.query(pts, mark_accessed=mark_accessed) for lv in self.levels]
        return torch.cat(per, -1)

    # utils -------------------------------------------------------------------
    @torch.no_grad()
    def collision_stats(self):
        return {f"level_{i}": lv.collision_stats() for i, lv in enumerate(self.levels)}

    @torch.no_grad()
    def get_accessed_indices(self):
        return [lv.get_accessed_indices() for lv in self.levels]

    @torch.no_grad()
    def reset_access_log(self):
        for lv in self.levels:
            lv.reset_access_log()

    def get_scene_bounds(self):
        """Return the scene bounds."""
        return self.scene_bound_min, self.scene_bound_max



    def query_feature(self, x: torch.Tensor, scene_id: torch.Tensor, mark_accessed: bool = True) -> torch.Tensor:
        assert scene_id.unique().numel() == 1, "VoxelHashTable can only handle one scene_id"
        return self.query_voxel_feature(x, mark_accessed=mark_accessed)
