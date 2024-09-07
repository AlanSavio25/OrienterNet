from typing import Tuple

import torch

from maploc.utils.wrappers import Transform2D

def build_query_grid(f_bev):
    """Computes coordinates for each pixel in f_bev"""

    h, w = f_bev.shape[-2:]  # h = 129, w = 64
    bev_t_cam = torch.tensor([(h - 1) / 2, 0.0])

    bev_ij_pts = (
        torch.stack(torch.unravel_index(torch.arange(h * w), (h, w)), -1)
        - bev_t_cam
    )

    # BEV faces east in the map frame by default, so we rotate the coords by 90deg
    bev_ij_pts = (
        Transform2D.from_degrees(torch.tensor([-90]), torch.zeros(2)) @ bev_ij_pts
    )

    return bev_ij_pts.view(h, w, 2)

def neural_cutout(
    f_bev: torch.Tensor,  # I,J,2
    f_map: torch.Tensor,  # B,C,H,W
    map_T_cam: Transform2D,  # B
    padding_mode: str = "border",
    mode: str = "bilinear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cuts out a posed BEV from the Neural Map f_map."""

    bev_ij_pts = build_query_grid(f_bev)

    batch_size = map_T_cam.shape[0]
    bev_ij_pts_posed = map_T_cam @ bev_ij_pts.view(-1, 2).to(map_T_cam._data)
    bev_ij_pts_posed = bev_ij_pts_posed.view(bev_ij_pts.shape).repeat(
        batch_size, 1, 1, 1
    )
    map_size = torch.tensor(f_map.shape[-2:]).to(bev_ij_pts_posed)

    # Grid sample f_map at posed coords
    grid_bev = bev_ij_pts_posed.flip(-1)
    grid_bev = grid_bev * 2 / (map_size - 1) - 1.0  # Normalize to [-1, 1]
    valid = torch.all(
        (bev_ij_pts_posed >= 0) & (bev_ij_pts_posed < map_size), -1
    )  # (B, 129, 64)
    f_bev_cutout = torch.nn.functional.grid_sample(
        f_map, grid_bev, mode, padding_mode, align_corners=True
    )

    return f_bev_cutout, valid
