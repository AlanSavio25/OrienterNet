from dataclasses import dataclass
from typing import Tuple

import torch

from maploc.utils.wrappers import Transform2D


@dataclass
class GridND:
    """N-dimensional regular grid.

    extent: Number of cells along each dimension
    cell_size: Physical size of each cell, in meters
    num_cells: Total number of cells
    extent_meters: Physical size of the grid, in meters
    """

    extent: Tuple[int, ...]
    cell_size: float

    @classmethod
    def from_extent_meters(cls, extent_meters: Tuple[float, ...], cell_size: float):
        extent = tuple(i / cell_size for i in extent_meters)
        if not all(e % 1 == 0 for e in extent):
            raise ValueError(
                f"The metric grid extent {extent_meters} is not divisble "
                f"by the cell size {cell_size}"
            )
        return cls(tuple(map(int, extent)), cell_size)

    def xyz_to_index(self, xyz):
        return torch.floor(xyz / self.cell_size).int()

    def index_to_xyz(self, idx):
        return (idx + 0.5) * self.cell_size

    @property
    def num_cells(self) -> int:
        return torch.prod(self.extent)

    @property
    def extent_meters(self):
        return torch.tensor(self.extent) * self.cell_size

    def index_in_grid(self, idx):
        return ((idx >= 0) & (idx < torch.tensor(self.extent))).all(-1)

    def xyz_in_grid(self, xyz):
        return ((xyz >= 0) & (xyz < self.extent_meters)).all(-1)

    def grid_index(self):
        grid = torch.stack(
            torch.meshgrid([torch.arange(e) for e in self.extent], indexing="ij")
        )
        return torch.movedim(grid, 0, -1)


@dataclass
class Grid2D(GridND):
    """2-dimensional regular grid"""

    extent: Tuple[int, int]


class Grid3D(GridND):
    """3-dimensional regular grid"""

    extent: Tuple[int, int, int]

    def bev(self) -> Grid2D:
        return Grid2D(self.extent[:2], self.cell_size)


def interpolate_nd(
    array: torch.Tensor,  # dim, H, W
    points: torch.Tensor,  # N,2 in ij indexing
    valid_array: torch.Tensor = None,
    padding_mode: str = "zeros",
    mode: str = "bilinear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate an N-dimensional array at the given points."""

    size = torch.tensor(array.shape[-2:]).to(points)  # H, W
    valid_bounds = torch.all((points >= 0) & (points < size), -1).squeeze()
    grid_pts = (points * 2) / (size - 1).clamp(min=1) - 1.0
    grid_pts = grid_pts.flip(-1)  # grid_sample assumes xy indexing.
    values = torch.nn.functional.grid_sample(
        array[None, ...],
        grid_pts[None, None, ...],
        mode,
        padding_mode,
        align_corners=True,  # sample from center of cell
    ).squeeze()

    valid_mask = None
    if valid_array is not None:
        # Excludes bev points that fall in invalid map regions from pose scoring
        nan_mask = torch.where(valid_array, 0, torch.nan)
        nan_points_mask = torch.nn.functional.grid_sample(
            nan_mask[None, None, ...],
            grid_pts[None, None, ...],
            mode,
            padding_mode,
            align_corners=True,
        )
        # valid = valid & ~torch.isnan(nan_points_mask)
        valid_mask = ~torch.isnan(nan_points_mask).squeeze()

    return values, valid_bounds, valid_mask


def interpolate_score_maps_orienternet(  # rename to something like score_pose
    points: torch.Tensor,  # I*J, 2
    f_map: torch.Tensor,  # 8, H, W
    f_bev: torch.Tensor,  # 8, I, J
    valid_map: torch.Tensor,  # H, W
):
    # Interpolate f_map at these points
    map_interp, valid_interp_bounds, valid_interp_mask = interpolate_nd(
        f_map, points, valid_map  # 8, H, W  # I*J, 2
    )
    scores_interp = torch.sum(map_interp * f_bev.reshape(map_interp.shape), dim=0)
    return scores_interp, valid_interp_bounds, valid_interp_mask


def pose_scoring_orienternet(
    map_T_cam: torch.Tensor,  # 1, 3 (single pose)
    f_map: torch.Tensor,  # 8, H, W
    f_bev: torch.Tensor,  # 8, I, J
    bev_ij_pts: torch.Tensor,  # I, J, 2
    valid_bev: torch.Tensor,  # I, J
    valid_map: torch.Tensor,  # H, W
    mask_out_of_bounds: bool = True,
    mask_mapmask: bool = False,
):
    """Compute a consistency score for a given pose"""

    map_T_cam = Transform2D(map_T_cam)
    bev_ij_pts_posed = map_T_cam @ bev_ij_pts.reshape(-1, 2)  # flatten bev coords
    scores_points, valid_bounds, valid_mask = interpolate_score_maps_orienternet(
        bev_ij_pts_posed, f_map, f_bev, valid_map
    )

    mask_out_of_bounds = True
    if mask_out_of_bounds:
        # invalidates bev points that fall outside the HxW mask
        valid_bev = valid_bev & valid_bounds.reshape(*valid_bev.shape)

    if mask_mapmask:
        # invalidates bev points that fall in invalid map areas
        valid_bev = valid_bev & valid_mask.reshape(*valid_bev.shape)

    # Invalidate poses that fall in invalid map areas. # fixme
    # cam_origin_idx = (bev_ij_pts.shape[0] // 2, 0)
    # is_valid_pose = valid_mask.view(*bev_ij_pts.shape[:2])[cam_origin_idx]

    pose_score = torch.sum(valid_bev.reshape(-1) * scores_points)

    return pose_score


def grid_refinement_orienternet(
    map_T_cam_init: torch.Tensor,  # Single best Pose
    f_map: torch.Tensor,  # Single map 8,H,W,
    f_bev: torch.Tensor,  # single map N,8
    f_bev_pts: torch.Tensor,
    valid_bev,
    valid_map,
    delta_p,
    range_p,
    delta_r,
    range_r,
):
    """Score poses distributed on a grid centered at an initial pose"""

    p_vals = torch.arange(-range_p, range_p + delta_p, delta_p)
    r_vals = torch.arange(-range_r, range_r + delta_r, delta_r)

    grid_r, grid_pi, grid_pj = torch.meshgrid(r_vals, p_vals, p_vals, indexing="ij")

    offsets_rij = torch.stack([grid_r, grid_pi, grid_pj], dim=-1).view(-1, 3)

    cam_T_cam_offset = Transform2D.from_degrees(
        angle=offsets_rij[..., :1], t=offsets_rij[..., 1:]
    ).to(map_T_cam_init)
    map_T_cam_samples = Transform2D(map_T_cam_init) @ cam_T_cam_offset

    scores = pose_scoring_many_orienternet(  # many poses
        map_T_cam_samples._data,
        f_map,
        f_bev,
        f_bev_pts,  # facing east
        valid_bev,
        valid_map,
    )

    score_refined, best_idx = torch.max(scores, dim=-1)
    map_T_cam_refined = map_T_cam_samples[best_idx[None]][0]._data
    scores = scores.reshape(grid_r.shape, -1)

    return map_T_cam_refined, score_refined, map_T_cam_samples._data, scores


pose_scoring_many_orienternet = torch.vmap(
    pose_scoring_orienternet, in_dims=(0,) + (None,) * 5
)
pose_scoring_many_orienternet_batched = torch.vmap(
    pose_scoring_many_orienternet, in_dims=(0,) * 2 + (None,) + (0,) * 2
)
grid_refinement_orienternet_batched = torch.vmap(
    grid_refinement_orienternet, in_dims=(0,) * 3 + (None,) * 1 + (0,) * 2 + (None,) * 4
)
