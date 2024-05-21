import numpy as np
import torch
from typing import Optional, Tuple, TypeVar, Type
from dataclasses import dataclass

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
                f'The metric grid extent {extent_meters} is not divisble '
                f'by the cell size {cell_size}'
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
        grid = torch.stack(torch.meshgrid([torch.arange(e) for e in self.extent], indexing='ij'))
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
    grid_pts = grid_pts.flip(-1) # grid_sample assumes xy indexing.
    # torch where size == 0, grid_pts -1
    values = torch.nn.functional.grid_sample(
        array[None, ...],
        grid_pts[None, None, ...],
        mode,
        padding_mode,
        align_corners=True, # sample from center of cell
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