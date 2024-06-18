import functools
from typing import Tuple

import torch

from maploc.models.mlp import MLP
from maploc.utils import grids
from maploc.utils.wrappers import Camera, Transform2D, Transform3D

from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjectionDepth
from .voting import TemplateSampler


@functools.partial(torch.vmap, in_dims=(0, 0, 0))
def project_points_to_view(
    cam_R_gcam: torch.Tensor, camera: torch.Tensor, points: torch.Tensor  # 3x3
):
    """Project a set of points to a view"""
    camera = Camera(camera)  # vmap doesn't allow Camera input

    gcam_R_world = torch.eye(3).to(cam_R_gcam)
    gcam_R_world[1, 1], gcam_R_world[1, 2] = 0, -1
    gcam_R_world[2, 1], gcam_R_world[2, 2] = 1, 0

    # TODO: replace T with R
    cam_T_gcam = Transform3D.from_Rt(cam_R_gcam, torch.zeros((3)).to(cam_R_gcam))
    gcam_T_world = Transform3D.from_Rt(gcam_R_world, torch.zeros((3)).to(cam_R_gcam))
    points_view = cam_T_gcam.float() @ gcam_T_world.float() @ points
    depth = points_view[..., -1]
    distance = torch.linalg.norm(points_view, dim=-1, keepdim=True)
    rays = points_view / distance.clip(min=1e-5)
    p2d, visible = camera.world2image(points_view)
    p2d = p2d.flip(-1)  # xy to ij indexing
    return p2d, visible, depth, rays


@functools.partial(torch.vmap, in_dims=(0, 0))  # map over batch
def interpolate_features(array: torch.Tensor, points: torch.Tensor):
    """Interpolation of features at projection of grid points"""
    interp, _, _ = grids.interpolate_nd(array, points, padding_mode="border")
    return interp


@functools.partial(torch.vmap, in_dims=(0, 0, None))  # map over batch
@functools.partial(torch.vmap, in_dims=(0, 0, None))  # map over points
def interpolate_depth_scores(
    score_depths: torch.Tensor, depth: float, depth_min_max: Tuple[float, float]
):
    """Interpolate a 1D depth distribution at point reprojections"""
    num_bins = score_depths.shape[-1]
    dmin, dmax = depth_min_max
    depth = depth.clamp(dmin, dmax)
    t = torch.log(depth / dmin) / torch.log(dmax / dmin)
    index = 0.5 + t * (num_bins - 1)  # map [0, 1] to [0.5, num_bins-0.5=32.5]
    index = index[None, None]
    grid_pts = torch.cat((torch.zeros_like(index), index), dim=-1)
    score_point, _, _ = grids.interpolate_nd(
        score_depths[None, None, :], grid_pts, padding_mode="border"
    )

    return score_point


@functools.partial(torch.vmap, in_dims=(0, 0, None, 0))  # map over batch
@functools.partial(torch.vmap, in_dims=(0, 0, None, None))  # map over points
def interpolate_scale_scores(
    score_scales: torch.Tensor,
    depth: float,
    scale_min_max: Tuple[float],
    camera: torch.Tensor,
):
    """Interpolate a 1D scale distribution at point reprojections"""
    camera = Camera(camera)
    scale = camera.f[1] / depth
    num_bins = score_scales.shape[-1]

    smin, smax = scale_min_max  # smin and smax are log-scales
    log_scale = torch.log2(scale)
    log_scale = log_scale.clamp(smin, smax)
    t = (log_scale - smin) / (smax - smin)
    index = t * (num_bins - 1)
    index = index[None, None]
    grid_pts = torch.cat((torch.zeros_like(index), index), dim=-1)
    score_point, _, _ = grids.interpolate_nd(
        score_scales[None, None, :], grid_pts, padding_mode="border"
    )

    return score_point


def build_frustum_grid(cell_size: float, depth: float, width: float):
    #    hfov_deg: float = None):
    """Build a gravity-aligned grid bounding the camera frustum"""
    # grid[x,y] gives us the x y coordinate in bev coordinate frame

    # width =  2 * depth * np.tan(np.deg2rad(hfov_deg/2))
    # width = np.floor(width)

    grid = grids.Grid2D.from_extent_meters((width, depth), cell_size)
    grid_t_cam = torch.tensor([width / 2, 0.0])

    grid_xy_pts = grid.index_to_xyz(grid.grid_index())
    cam_xy_pts = grid_xy_pts - grid_t_cam
    return grid, grid_t_cam, cam_xy_pts, grid_xy_pts


def build_query_grid():
    """Computes cam coordinates for each pixel in f_bev"""
    # This function is currently being used by neural_map_cutout and
    # ransac_matching. todo: replace with build_frustum_grid
    # bev_ij_pts is of shape 129x64x2, where the camera center is at
    # bev_ij_pts[64,0]. So bev_ij_pts[64,0] = [0,0]

    # h, w = f_bev.shape[-2:]  # h = 129, w = 64
    h = 129
    w = 64
    bev_t_cam = torch.tensor([(h - 1) / 2, 0.0])
    bev_ij_pts = (
        torch.stack(torch.unravel_index(torch.arange(h * w), (h, w)), -1) - bev_t_cam
    )
    # BEV faces east in the map frame by default, so we rotate the coords by 90deg
    bev_ij_pts = (
        Transform2D.from_degrees(torch.tensor([-90]), torch.zeros(2)) @ bev_ij_pts
    )
    return bev_ij_pts.view(h, w, 2)


class VerticalPooling(BaseModel):
    """Flatten a 3D feature grid into a 2D BEV by pooling vertically."""

    default_conf = {
        "pooling": "max",
    }

    def _init(self, conf):
        self.pooling_ops = {k: getattr(torch, k) for k in ("max", "mean", "sum")}

    def _forward(self, data):
        valid_any = data["valid"].any(-1)
        features = self.pooling_ops[self.conf.pooling](data["features"], dim=-2)
        if self.conf.pooling == "max":
            features = features[0]
        features = torch.where(valid_any[..., None], features, 0.0)
        return {"features": features, "valid": valid_any}


class BEVMapper(BaseModel):
    """Predict a top-down Bird's-Eye-View feature plane from a single image."""

    default_conf = {
        "image_encoder": "???",
        "scale_classifier": "linear",  # or 'mlp'
        "scale_mlp": None,
        "fusion_mlp": None,  # only for "inverse" mode
        # At f=256px, logscale 0 corresponds to 256m depth, -1 -> 512m. 9 -> 0.5m
        "scale_range": [-1, 9],
        "mode": "forward",  # or "inverse"
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_scale_bins": "???",
        "num_rotations": "???",
        "bev_net": "???",
        "latent_dim": "???",
        "grid_height": 12,
        "grid_cell_size": 0.5,
        "grid_z_offset": 0,
        "grid_z_offset_range": [-1, 1],
        "feature_depth_fusion": "softmax",
        "vertical_pooling": "mean",
    }

    def _init(self, conf):

        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        self.image_encoder = Encoder(conf.image_encoder.backbone)
        ppm = conf.pixel_per_meter
        self.bev_ij_pts = None  # TODO: get rid of this

        self.projection_polar = PolarProjectionDepth(
            conf.z_max,
            ppm,
            conf.scale_range,
            conf.z_min,
        )
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations
        )

        if conf.mode == "inverse":
            if conf.feature_depth_fusion == "mlp":
                # Fuse feature and depth score to predict a new feature
                self.fusion_mlp = MLP(
                    conf.fusion_mlp
                )  # input_dim: feat_dim+score, out:feat_dim
            # TODO: rename z_max to max_depth?
            self.grid, self.grid_t_cam, self.cam_xy_pts, self.grid_xy_pts = (
                build_frustum_grid(
                    cell_size=conf.grid_cell_size,
                    depth=conf.z_max,
                    width=conf.x_max * 2 + conf.grid_cell_size,
                )
            )

        elif conf.mode != "forward":
            raise ValueError(
                f"BEV mapper mode must be either 'forward' (OrienterNet) or  \
                'inverse' (SNAP). Got: {self.conf.mode}"
            )

        if conf.scale_classifier == "linear" or conf.mode == "forward":
            self.scale_classifier = torch.nn.Linear(
                conf.latent_dim, conf.num_scale_bins
            )
        elif conf.scale_classifier == "mlp":
            assert conf.scale_mlp is not None
            self.scale_classifier = MLP(conf.scale_mlp)

        self.vertical_pooling = VerticalPooling({"pooling": conf.vertical_pooling})

        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)
        if conf.bev_net is None:
            self.feature_projection = torch.nn.Linear(
                conf.latent_dim, conf.matching_dim
            )

    def _forward(self, data):

        pred = {}

        # Extract image features.
        level = 0
        f_image = self.image_encoder(data)["feature_maps"][level]
        camera = data["camera"].scale(1 / self.image_encoder.scales[level])
        camera = camera.to(data["image"].device, non_blocking=True)

        pred["pixel_scales"] = scales = self.scale_classifier(
            f_image.moveaxis(1, -1)
        )  # if snap, then this should be an mlp

        if self.conf.mode == "forward":

            # Map image columns to polar ray features
            f_polar = self.projection_polar(f_image, scales, camera)

            # Polar to cartesian
            with torch.autocast("cuda", enabled=False):
                f_bev, valid_bev, _ = self.projection_bev(
                    f_polar.float(), None, camera.float()
                )

            pred_bev = {}

            # Cartesian features through BEV Net -> f_bev+confidence
            if self.conf.bev_net is None:
                # channel last -> classifier -> channel first
                f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
                pred["bev"] = {"output": f_bev}
            else:
                pred_bev = pred["bev"] = self.bev_net(
                    {"input": f_bev}
                )  # SNAP: This can probably be reused for the SNAP implementation
                f_bev = pred_bev["output"]

            # Refactoring: convert bev to memory layout
            f_bev = pred["bev"]["output"] = torch.rot90(
                f_bev, -1, dims=(-2, -1)
            )  # B, C, I, J
            confidence_bev = pred_bev.get("confidence")  # B, I, J
            if confidence_bev is not None:
                confidence_bev = pred["bev"]["confidence"] = torch.rot90(
                    confidence_bev, -1, dims=(-2, -1)
                )
            valid_bev = pred["bev"]["valid_bev"] = torch.rot90(
                valid_bev, -1, dims=(-2, -1)
            )  # B, I, J

        elif self.conf.mode == "inverse":  # SNAP's BEV

            tile_T_cam = data["tile_T_cam"]
            cam_R_gcam = data["cam_R_gcam"]

            # Build 3D grid in front of camera. The coordinates are gcam coordinates

            # cam_xy_pts are 2d grid points centered around the camera
            # grid_xy_pts are 2d grid points centered at the bev origin
            xy = self.cam_xy_pts  # h,w,2

            if len(xy.shape) != 4:
                xy = xy[None].repeat_interleave(tile_T_cam.shape[0], dim=0)

            z_offset = -torch.tensor(4)
            # Add noise to the z_offset
            if self.conf.grid_z_offset_range is not None:
                z_min, z_max = self.conf.grid_z_offset_range  # -2, +2
                z_offset = z_offset + (
                    torch.rand(1).squeeze() * (z_max - z_min) + z_min
                )

            grid_height = self.conf.grid_height
            z_delta = 0.5
            z = torch.arange(0, grid_height, z_delta) + z_offset + z_delta / 2
            xy, z = torch.broadcast_tensors(
                xy[:, :, :, None, :], z[None, None, None, :, None]
            )
            xyz = torch.cat([xy, z[..., :1]], dim=-1).to(camera.device)

            grid_shape = xyz.shape[:-1]
            xyz_flat = xyz.reshape(len(xyz), -1, 3)  # B, N=129x64x24, 3

            # Compute the locations of 2D observations in camera view for all points
            p2d_view, visible, depth, _ = project_points_to_view(
                cam_R_gcam, camera._data, xyz_flat
            )

            # Plot projected points on image
            # image = torch.nn.functional.interpolate(data['image'],
            #                                         scale_factor=0.5,
            #                                         mode='bilinear'
            #                                         )[0].permute(1, 2, 0).cpu().numpy()
            # import matplotlib.pyplot as plt
            # p2d_view_np = p2d_view[0][visible[0]].clone().cpu().numpy()
            # plt.imshow(image)
            # plt.scatter(p2d_view_np[:, 1], p2d_view_np[:, 0], s=1, c='red')
            # plt.axis('off')
            # plt.savefig('image_with_visible_points.png')

            # Concat image features with scale scores and then interpolate
            f_proj = interpolate_features(
                torch.cat([f_image, scales.moveaxis(-1, -3)], 1), p2d_view
            )
            f_proj = f_proj.moveaxis(-1, -2)
            f_proj, scores_scales = f_proj.split(self.conf.latent_dim, dim=-1)

            # d_min_max = torch.tensor([self.conf.z_min, self.conf.z_max]).to(scores_scales)
            # scores_proj = interpolate_depth_scores(scores_scales, depth, d_min_max).moveaxis(1,0)
            s_min_max = torch.tensor(self.conf.scale_range).to(scores_scales)
            scores_proj = interpolate_scale_scores(
                scores_scales, depth, s_min_max, data["camera"]._data
            ).moveaxis(1, 0)
            scores_proj = scores_proj.moveaxis(-1, -2)

            grid_shape = (-1, *xyz.shape[-4:-1])

            if self.conf.feature_depth_fusion == "mlp":  # like snap
                # as in SNAP: X = MLP([f_proj, score]). Then, vertical pool to get M = max X
                f_grid = self.fusion_mlp(
                    torch.cat([f_proj, scores_proj[..., None]], dim=-1)
                )
            elif self.conf.feature_depth_fusion == "softmax":  # like orienternet
                scores_proj = scores_proj.reshape(*grid_shape, 1)
                vertical_softmax = torch.nn.Softmax(dim=-2)(scores_proj).reshape(
                    f_proj.shape[:-1]
                )
                f_grid = f_proj * vertical_softmax[..., None]
                # these features can now be vertically pooled using mean

            f_grid = torch.where(visible[..., None], f_grid, 0)

            # Reshape to 3D volume
            f_grid = f_grid.reshape(*grid_shape, f_grid.shape[-1])
            valid = visible.reshape(grid_shape)

            bev = self.vertical_pooling({"features": f_grid, "valid": valid})
            f_bev, valid_bev = bev["features"], bev["valid"]

            # Enforce a BEV shape of [129,64].
            # This allows us to generate a finer larger BEV and then downsample for memory efficiency
            if self.conf.grid_cell_size != 1 / self.conf.pixel_per_meter:
                h = int((self.conf.x_max * 2 * self.conf.pixel_per_meter) + 1)
                w = int(self.conf.z_max * self.conf.pixel_per_meter)
                f_bev = torch.nn.functional.interpolate(
                    f_bev.moveaxis(-1, -3),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).moveaxis(-3, -1)
                nan_mask = torch.where(valid_bev, 0, torch.nan)
                nan_mask = torch.nn.functional.interpolate(
                    nan_mask.unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                valid_bev = ~torch.isnan(nan_mask)

            assert list(f_bev.shape[-3:-1]) == [129, 64] == list(valid_bev.shape[-2:])

            # Forward pooled BEV features through BEV Net
            # SNAP doesn't use a bev_net. Worth checking if it helps
            if self.conf.bev_net is None:
                # channel last -> classifier -> channel first
                f_bev = self.feature_projection(f_bev).moveaxis(-1, 1)
                pred["bev"] = {"output": f_bev}
            else:
                pred_bev = pred["bev"] = self.bev_net({"input": f_bev.moveaxis(-1, 1)})

                f_bev = pred_bev["output"]

            pred["bev"]["valid_bev"] = valid_bev

        pred = {**pred, "features_image": f_image}
        return pred
