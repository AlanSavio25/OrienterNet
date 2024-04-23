from typing import Tuple
import torch
import numpy as np
import random

from maploc.utils.wrappers import Transform2D


def sample_transforms_ransac(
    bev_ij_pts: torch.Tensor, # IxJx2
    prob_points: torch.Tensor, # NxHxW
    num_poses: int = 5_000,
    num_retries: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly sample poses derived from the most confident correspondences"""

    shape = prob_points.shape
    prob_points = prob_points.reshape(-1)

    num_obs = 2  # num of observations per pose sample

    # Sample correspondences
    p = prob_points.cumsum(0)
    sample_values = p[-1] * (1 - torch.rand(num_poses * num_retries * num_obs)).to(p)
    indices = torch.searchsorted(p, sample_values)

    indices = torch.stack(torch.unravel_index(indices, shape), -1)
    pool_shape = (num_poses, num_retries, num_obs, 2)

    bev_ij_pts_pool = bev_ij_pts.reshape(-1, 2)[indices[..., 0]].reshape(pool_shape).float()
    map_ij_pts_pool = indices[..., 1:].reshape(pool_shape).float()

    if num_retries > 1:
        # sample multiple minimal sets and retain those that are most consistent
        d_bev = torch.linalg.norm(torch.diff(bev_ij_pts_pool, dim=-2).squeeze(-2), dim=-1) # [num_poses, num_retries] 
        d_map = torch.linalg.norm(torch.diff(map_ij_pts_pool, dim=-2).squeeze(-2), dim=-1) # [num_poses, num_retries]
        ratio = torch.maximum(d_bev / d_map.clamp(min=1e-5), d_map / d_bev.clamp(min=1e-5)) # [num_poses, num_retries]
        select_indices = torch.argmin(ratio, dim=-1)
        select_fn = torch.vmap(lambda x, i: x[i[None]][0]) # work-around for .item() error
        bev_ij_pts_pool = select_fn(bev_ij_pts_pool, select_indices)
        map_ij_pts_pool = select_fn(map_ij_pts_pool, select_indices)
    else:
        map_ij_pts_pool = map_ij_pts_pool.squeeze(1)
        bev_ij_pts_pool = bev_ij_pts_pool.squeeze(1)

    map_R_cam, map_t_cam, _, _ = torch.vmap(kabsch_2d)(
        bev_ij_pts_pool, map_ij_pts_pool
    )

    map_T_cam_samples = Transform2D.from_Rt(map_R_cam, map_t_cam)._data #vmap doesn't allow TensorWrappers

    return map_T_cam_samples, bev_ij_pts_pool, map_ij_pts_pool


def kabsch_2d(
    i_pts: torch.Tensor, j_pts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the least-squares 2D transform between two sets of points.
    """

    mu_i = i_pts.mean(0)
    mu_j = j_pts.mean(0)
    i_pts_centered = i_pts - mu_i
    j_pts_centered = j_pts - mu_j

    cov = i_pts_centered.T @ j_pts_centered
    u, s, vh = torch.svd(cov)

    sign = torch.sign(torch.det(u @ vh.T))
    u[..., -1] *= sign
    s[..., -1] *= sign
    valid = s[1] > 1e-16 * s[0]

    i_r_j = vh @ u.T
    i_t_j = mu_j - i_r_j @ mu_i

    i_pts_aligned = (i_r_j @ i_pts.T).T + i_t_j
    rssd = torch.sqrt(((i_pts_aligned - j_pts) ** 2).sum())
    return i_r_j, i_t_j, valid, rssd


def interpolate_nd(
    array: torch.Tensor, # H, W, 1
    points: torch.Tensor, # N D
    valid_array: torch.Tensor,
    padding_mode: str = "zeros",
    mode: str = "bilinear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate an N-dimensional array at the given points."""

    size = torch.tensor(array.shape[-2:]).to(points)  # H, W
    valid_bounds = torch.all((points >= 0) & (points < size), -1)
    grid_pts = points.flip(-1)
    grid_pts = (grid_pts * 2) / (size - 1) - 1.
    values = torch.nn.functional.grid_sample(array[None, ...], 
                                             grid_pts[None, None, ...], 
                                             mode, 
                                             padding_mode, 
                                             align_corners=True)

    if valid_array is not None:
        # Excludes bev points that fall in invalid map regions from pose scoring
        nan_mask = torch.where(valid_array, 0, torch.nan)
        nan_points_mask = torch.nn.functional.grid_sample(nan_mask[None, None, ...],
                                                          grid_pts[None, None, ...],
                                                          mode,
                                                          padding_mode,
                                                          align_corners=True)
        # valid = valid & ~torch.isnan(nan_points_mask)
        valid_mask = ~torch.isnan(nan_points_mask)

    return values.squeeze(), valid_bounds.squeeze(), valid_mask.squeeze()


def interpolate_score_maps(
    points: torch.Tensor, # I*J, 2
    scores: torch.Tensor,  # I*J, H, W
    valid_map: torch.Tensor,  # H, W
):
    interp_many = torch.vmap(interpolate_nd, in_dims=(0, 0, None))

    scores_interp, valid_interp_bounds, valid_interp_mask = interp_many(
        scores[..., None, :, :], # add channel dim so we have I*J, 1, H, W
        points[..., None, :], # I*J, 1, 2
        valid_map,
    )

    return scores_interp, valid_interp_bounds, valid_interp_mask 


def pose_scoring(
    map_T_cam: torch.Tensor,  # 1, 3 (single pose)
    sim_points: torch.Tensor,  # N, H, W
    bev_ij_pts: torch.Tensor,  # I, J, 2
    valid_bev: torch.Tensor,  # I, J
    valid_map: torch.Tensor,  # H, W
    mask_out_of_bounds: bool = True,
    mask_mapmask: bool = False
):
    """Compute a consistency score for a given pose"""

    map_T_cam = Transform2D(map_T_cam)
    bev_ij_pts_posed = map_T_cam @ bev_ij_pts.reshape(-1, 2)
    scores_points, valid_bounds, valid_mask = interpolate_score_maps(
        bev_ij_pts_posed, sim_points, valid_map
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


def grid_refinement(
        map_T_cam_init: torch.Tensor, # tensorized Transform2D
        sim_points: torch.Tensor, # NxHxW
        f_bev: torch.Tensor, # Nx2
        valid_bev: torch.Tensor, # N
        valid_map: torch.Tensor, # HxW
        ):
    """Score poses distributed on a regular grid centered at an initial pose"""

    delta_p = 0.75 # reduced for memory
    delta_r = 0.75
    range_p = 6.
    range_r = 6.

    p_vals = torch.arange(-range_p, range_p + delta_p, delta_p)
    r_vals = torch.arange(-range_r, range_r + delta_r, delta_r)

    
    grid_r, grid_pi, grid_pj = torch.meshgrid(r_vals, p_vals, p_vals, indexing="ij")

    offsets_rij = torch.stack([grid_r, grid_pi, grid_pj], dim=-1).view(-1, 3)

    cam_T_cam_offset = Transform2D.from_degrees(angle=offsets_rij[..., :1], t=offsets_rij[..., 1:]).to(map_T_cam_init)
    map_T_cam_samples = Transform2D(map_T_cam_init) @ cam_T_cam_offset

    scores = pose_scoring_many(map_T_cam_samples._data, #vmap does not support TensorWrappers
                               sim_points,
                               f_bev,
                               valid_bev,
                               valid_map
                               )

    score_refined, best_idx = torch.max(scores, dim=-1)
    map_T_cam_refined = map_T_cam_samples[best_idx[None]][0]._data
    scores = scores.reshape(grid_r.shape, -1)

    return map_T_cam_refined, score_refined, scores


def test_kabsch_2d():
    pass_count = 0
    valid_count = 0
    total = 50
    for _ in range(total):

        A = torch.tensor([[0.0, 1.0], [2.0, 3.0], [10.5, 0.5]], dtype=torch.float)
        A += torch.rand_like(A)
        angle0 = torch.deg2rad(torch.randn(1) * 350)
        R_gt = torch.tensor(
            [
                [torch.cos(angle0), -torch.sin(angle0)],
                [torch.sin(angle0), torch.cos(angle0)],
            ],
            dtype=torch.float,
        )
        B = (R_gt @ A.T).T
        t_gt = torch.tensor([5.0, 1.0])
        B = B + t_gt
        R, t, valid, rssd = kabsch_2d(A, B)
        A_aligned = (R @ A.T).T + t
        rssd = torch.sqrt(((A_aligned - B) ** 2).sum())
        if rssd < 1e-5:
            pass_count += 1
        if valid:
            valid_count += 1
    print(f"Tests passed: {((pass_count / total)*100):.2f}%")
    print(f"Valid: {((valid_count / total)*100):.2f}%")


sample_transforms_ransac_batched = torch.vmap(
    sample_transforms_ransac, in_dims=(None,)* 1 + (0,) * 1 + (None,) * 2, randomness="different"
)
pose_scoring_many = torch.vmap(pose_scoring, in_dims=(0,) + (None,) * 4)
pose_scoring_many_batched = torch.vmap(pose_scoring_many, in_dims=(0,) * 2 + (None,) + (0,)*2)
grid_refinement_batched  = torch.vmap(grid_refinement, in_dims=(0,) * 2 + (None,) * 2 + (0,) * 1)