"""Interface for OrienterNetv2 inference"""

print("starting to infer")
from maploc.osm.tiling import TileManager
from maploc.utils.viz_localization import (
    likelihood_overlay,
    plot_dense_rotations,
    add_circle_inset,
)
from maploc.utils.viz_2d import features_to_RGB
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images


from copy import deepcopy
from omegaconf import DictConfig, OmegaConf, read_write

from typing import Optional, Tuple, Dict, Union

from maploc.utils.viz_localization import plot_pose, plot_bev
from maploc.utils.wrappers import Transform2D
from maploc.utils import grids
import numpy as np
import torch
from pathlib import Path

from . import logger
from .data.image import pad_image, rectify_image, resize_image
from .models.orienternet import OrienterNet
from .models.voting import argmax_xyr, fuse_gps, log_softmax_spatial
from .osm.raster import Canvas
from .utils.exif import EXIF
from .utils.geo import BoundaryBox, Projection
from .utils.io import read_image
from .utils.wrappers import Camera, Transform2D
from scipy.spatial.transform import Rotation

from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_utilities.core.apply_func import apply_to_collection
import matplotlib.pyplot as plt
from .utils.viz_2d import features_to_RGB, plot_images, save_plot

from .evaluation.run import disable_key

try:
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="orienternet")
except ImportError:
    geolocator = None

try:
    from gradio_client import Client, handle_file

    # logger.info("Loading GeoCalib model...")
    # calibrator = Client("veichta/GeoCalib")
    # calibrator = torch.hub.load("cvg/GeoCalib", "GeoCalib", trust_repo=True, verbose=True)
    calibrator = None
except (ImportError, ValueError):
    calibrator = None


def image_calibration(image_path):
    """Estimate camera intrinsics and gravity vector using GeoCalib"""
    result = calibrator.predict(
        handle_file(image_path),
        camera_model="simple_radial",
        plot_up=False,
        plot_up_confidence=False,
        plot_latitude=False,
        plot_latitude_confidence=False,
        plot_undistort=False,
        api_name="/process_results",
    )
    params = [
        float(line.split(":")[1].split("°")[0].strip())
        for line in result[0].split("\n")[1:4]
    ]
    return params


def camera_from_exif(exif: EXIF, vfov: Optional[float] = None) -> Camera:
    w, h = image_size = exif.extract_image_size()
    _, f_ratio = exif.extract_focal()
    if f_ratio == 0:
        if vfov is not None:
            f = h / 2 / np.tan(np.deg2rad(vfov) / 2)
        else:
            return None
    else:
        f = f_ratio * max(image_size)
    return Camera.from_dict(
        dict(
            model="SIMPLE_PINHOLE",
            width=w,
            height=h,
            params=[f, w / 2 + 0.5, h / 2 + 0.5],
        )
    )


def preprocess_inputs(
    image_path: str,
    prior_latlon: Optional[Tuple[float, float]] = None,
    prior_address: Optional[str] = None,
    fov: Optional[float] = None,
    tile_size_meters: int = 64,
):
    """Read image, estimate camera calibration, prepare map tile"""

    image = read_image(image_path)
    with open(image_path, "rb") as fid:
        exif = EXIF(fid, lambda: image.shape[:2])

    latlon = None
    if prior_latlon is not None:
        latlon = prior_latlon
        logger.info("Using prior latlon %s.", prior_latlon)
    if prior_address is not None:
        if geolocator is None:
            raise ValueError("geocoding unavailable, install geopy.")
        location = geolocator.geocode(prior_address)
        if location is None:
            logger.info("Could not find any location for address '%s.'", prior_address)
        else:
            logger.info("Using prior address '%s'", location.address)
            latlon = (location.latitude, location.longitude)
    if latlon is None:
        geo = exif.extract_geo()
        if geo:
            alt = geo.get("altitude", 0)  # read if available
            latlon = (geo["latitude"], geo["longitude"], alt)
            logger.info("Using prior location from EXIF.")
        else:
            logger.info("Could not find any prior location in the image EXIF metadata.")
    if latlon is None:
        raise ValueError(
            "No location prior given or found in the image EXIF metadata: "
            "maybe provide the name of a street, building or neighborhood?"
        )
    latlon = np.array(latlon)

    if calibrator is not None:
        roll, pitch, fov = image_calibration(image_path)
        logger.info("Using (roll, pitch, fov) %s, %s, %s.", roll, pitch, fov)
    else:
        roll, pitch = None, None
        logger.info("No estimated roll and pitch")

    # logger.info("Using cam_R_gcam %s.", cam_R_gcam)

    camera = camera_from_exif(exif, fov)
    if camera is None:
        raise ValueError(
            "No camera intrinsics found in the EXIF, provide a vertical FoV guess."
        )

    proj = Projection(*latlon)
    center = proj.project(latlon)
    bbox = BoundaryBox(center, center) + tile_size_meters
    return image, camera, (roll, pitch), proj, bbox  # , latlon


pretrained_models = dict(
    OrienterNet_MGL=("orienternet_mgl.ckpt", dict(num_rotations=256)),
    OrienterNetv2=("orienternetv2.ckpt", dict(num_rotations=256)),
    OrienterNetv2_with_satellite=(
        "orienternetv2_with_satellite.ckpt",
        dict(num_rotations=256),
    ),
    coarse_prior=("coarse_prior.ckpt", dict(num_rotations=256)),
)
EXP_PATH = "inference_experiments"


# TODO: remove all TODO
class OrienterNetv2:
    """Interface for model inference"""

    def __init__(
        self,
        experiment_or_path: Optional[str] = "OrienterNet_MGL",
        prior_exp_or_path: Optional[str] = "coarse_prior",
        device=None,
        **kwargs,
    ):

        if experiment_or_path in pretrained_models:
            path = pretrained_models[experiment_or_path][0]
        else:
            path = experiment_or_path

        path = EXP_PATH / Path(path)

        if not path.exists():
            logger.info(
                f"Ckpt does not exist at: {path}. Loading model from torch hub..."
            )
            path.parent.mkdir(exist_ok=True, parents=True)
            CHECKPOINT_URL = "https://github.com/AlanSavio25/OrienterNet/releases/download/releasev1.0/orienternetv2.ckpt"
            torch.hub.download_url_to_file(CHECKPOINT_URL, path)

        ckpt = torch.load(path, map_location=(lambda storage, loc: storage))
        config = ckpt["hyper_parameters"]
        config.model.update(kwargs)
        config.model.image_encoder.backbone.pretrained = False

        model = OrienterNet(config.model).eval()
        state = {k[len("model.") :]: v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(state, strict=True)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        self.model = model
        self.config = config
        self.device = device

        if prior_exp_or_path is not None:
            if prior_exp_or_path in pretrained_models:
                path = pretrained_models[prior_exp_or_path][0]
            else:
                path = prior_exp_or_path

            path = EXP_PATH / Path(path)

            if not path.exists():
                logger.info(
                    f"PRIOR ckpt does not exist at: {path}. Loading model from torch hub"
                )
                path.parent.mkdir(exist_ok=True, parents=True)
                CHECKPOINT_URL = "https://github.com/AlanSavio25/OrienterNet/releases/download/releasev1.0/coarse_prior.ckpt"
                torch.hub.download_url_to_file(CHECKPOINT_URL, path)

            p_ckpt = torch.load(path, map_location=(lambda storage, loc: storage))
            p_config = p_ckpt["hyper_parameters"]
            # p_config.model.update(kwargs)
            # p_config.model.image_encoder.backbone.pretrained = False

            p_model = OrienterNet(p_config.model).eval()
            state = {k[len("model.") :]: v for k, v in p_ckpt["state_dict"].items()}
            p_model.load_state_dict(state, strict=True)
            p_model = p_model.to(device)

            self.prior_model = p_model
            self.prior_config = p_config
        else:
            self.prior_model = None
            self.prior_config = None

        logger.info(
            f"Initialized OrienterNetv2 {'(with prior)' if self.prior_model is not None else ''}."
        )

    def prepare_inputs(
        self,
        image: np.ndarray,
        camera: Camera,
        proj,
        bbox,
        hierarchical=False,
        # canvas: Dict,
        # cam_R_gcam: torch.Tensor
        roll_pitch: Optional[Tuple[float]] = None,
    ):
        """Process image and prepare map raster for model inference"""

        logger.info(f"Preparing data for localization...")
        assert image.shape[:2][::-1] == tuple(camera.size.tolist())
        target_focal_length = self.config.data.resize_image / 2
        factor = target_focal_length / camera.f
        size = (camera.size * factor).round().int()

        image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
        valid = None
        # we don't need to rectify if we're using 3D grid projection. provide cam_R_gcam instead
        if roll_pitch is not None and None not in roll_pitch:  # todo: cleanup
            roll, pitch = roll_pitch
            R = Rotation.from_euler("ZX", (-roll, -pitch), degrees=True).as_matrix()
            R = torch.from_numpy(R)
            image, valid = rectify_image(image, camera.float(), cam_R_gcam=R)
        image, _, camera, *maybe_valid = resize_image(
            image, size.tolist(), camera=camera, valid=valid
        )
        valid = None if valid is None else maybe_valid

        # TODO: ensure this pads correctly.
        max_stride = max(self.model.bev_mapper.image_encoder.layer_strides)
        size = (torch.ceil(size / max_stride) * max_stride).int()
        image, valid, camera = pad_image(
            image, size.tolist(), camera, crop_and_center=True
        )

        if not hierarchical:
            ppm_list = self.config.data.pixel_per_meter
            zmax_list = self.config.data.z_max
        else:
            if self.prior_config is not None:
                ppm_list = self.prior_config.data.pixel_per_meter
                zmax_list = self.prior_config.data.z_max
            else:
                ppm_list = self.config.data.pixel_per_meter[-1:]  # coarsest ppm
                zmax_list = self.config.data.z_max[-1:]

        all_zmax = self.config.data.z_max + self.prior_config.data.z_max
        all_ppm = (
            self.config.data.pixel_per_meter + self.prior_config.data.pixel_per_meter
        )
        canvas = {key: None for key in all_zmax}
        tilers = {}

        for i, z_max in enumerate(all_zmax):
            ppm = all_ppm[i]
            tiler = TileManager.from_bbox(proj, bbox + 10, ppm)
            tilers[z_max] = tiler
            if z_max in zmax_list:
                canvas[z_max] = tiler.query(bbox)

        # for i, z_max in enumerate(zmax_list):
        #     ppm = ppm_list[i]
        #     tiler = TileManager.from_bbox(proj, bbox + 10, ppm)
        #     canvas[z_max] = tiler.query(bbox)

        semantic_map = {}
        for z_max, value in canvas.items():
            if value is None:
                semantic_map[z_max] = None
            else:
                semantic_map[z_max] = torch.from_numpy(value.raster).long()
                semantic_map[z_max] = torch.rot90(
                    semantic_map[z_max], -1, dims=(-2, -1)
                )

        return dict(
            image=image,
            canvas=canvas,
            tilers=tilers,
            semantic_map=semantic_map,
            roll_pitch=roll_pitch,
            cam_R_gcam=torch.eye(3),
            camera=camera.float(),
            valid=valid,
        )

    @torch.no_grad()
    def localize(self, data: dict, **kwargs):
        data_ = apply_to_collection(data, (torch.Tensor, Camera), lambda x: x[None])
        # data_ = {k: v.to(self.device)[None] for k, v in data.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            pred = self.model(data_)

        # Chain log_probs of 32m and 128m branches
        scores = [pred[k]["scores"].to("cpu") for k in self.config.data.z_max]

        fine_num_pixels = max([score_volume.shape[-2] for score_volume in scores])
        upsample_ppm = max(self.config.data.pixel_per_meter)
        h = w = fine_num_pixels
        scores = [
            torch.nn.functional.interpolate(
                score.moveaxis(-1, -3), size=(int(h), int(w)), mode="bilinear"
            ).moveaxis(-3, -1)
            for score in scores
        ]

        log_probs = [log_softmax_spatial(score) for score in scores]
        log_probs_chained = log_softmax_spatial(torch.stack(log_probs).sum(0))
        probs_chained = log_probs_chained.exp().cpu()
        del scores, log_probs

        uvr_max = argmax_xyr(log_probs_chained)
        ij_max = torch.flip(uvr_max[..., :2], dims=[-1])
        yaw_max = 180 - uvr_max[..., -1]
        map_T_max = Transform2D.from_degrees(yaw_max.unsqueeze(-1), ij_max)

        tile_T_cam_max_chained = (
            Transform2D.from_pixels(map_T_max, 1 / upsample_ppm)
        ).cpu()

        f_map_fine = pred[32.0]["features_map"].cpu()
        image = data["image"].cpu()  # padded/rectified image
        semantic_map = data["semantic_map"]  # this contains the memory-layout raster

        return (
            tile_T_cam_max_chained,
            map_T_max,
            probs_chained,
            f_map_fine,
            image,
            semantic_map,
            pred,
        )
        return

    @torch.no_grad()
    def localize_hierarchical(self, data: dict, topk: int, **kwargs):

        # data = self.prepare_inputs(image, camera, canvas, **kwargs)
        data = apply_to_collection(data, (torch.Tensor, Camera), lambda x: x[None])

        # decide whether prior should be used or not.
        # we use the prior only when the search region is too large.
        width = [v.bbox.size[0] for k, v in data["canvas"].items() if v is not None][0]
        min_threshold = 512  # m
        if width > min_threshold:
            use_prior = True
            p_ppm = self.prior_model.conf.pixel_per_meter[0]
            p_z = self.prior_model.conf.bev_mapper.z_max[0]
            assert (
                self.prior_model is not None
            ), f"Prior model required when Tile width ({width}) is > threshold {threshold}."
        else:
            use_prior = False

        f_ppm, c_ppm = self.model.conf.pixel_per_meter
        f_z, c_z = self.model.conf.bev_mapper.z_max

        # config
        num_k_coarse = topk if use_prior else 1
        num_k_fine = topk if not use_prior else 1
        csm_coarse = 256
        csm_fine = 64

        # pmap_crad is radius of a cmap in pmap coords
        # for masking - NMS
        pmap_crad = csm_coarse * p_ppm if use_prior else None
        cmap_frad = csm_fine * c_ppm

        topk_poses_fine = []
        p_topk_coords = []  # for plotting only

        logger.info(f"Running inference...")

        if use_prior:
            data_prior = disable_key(data, [f_z, c_z], remove_key=True)
            pred_prior = self.prior_model(data_prior)
            pscores = pred_prior[p_z]["scores"].clone()
            world_T_ptile = Transform2D.from_Rt(
                torch.eye(2), data_prior["canvas"][p_z].bbox.min_
            ).float()

        # This cache avoids recomputing network outputs for the same image
        bev_cache_fine = None
        bev_cache_coarse = None

        for k_idx_coarse in range(num_k_coarse):

            data_coarse = disable_key(data, [f_z])
            data_coarse = disable_key(data, [p_z], remove_key=True)
            if not use_prior:
                ccanvas = None
                craster = None
                bbox_ctile = data_coarse["canvas"][c_z].bbox
                # this is just for the metrics calculation, and must be removed
                # ctile_T_cam = data_coarse["tile_T_cam"][c_z].cpu()

                world_T_ctile = Transform2D.from_Rt(
                    torch.eye(2), bbox_ctile.min_
                ).float()
            else:
                if k_idx_coarse == 0:
                    pmap_T_maxcam = pred_prior[p_z]["map_T_cam_max"].float().squeeze(0)
                    ptile_T_maxcam = pred_prior[p_z]["tile_T_cam_max"].float()
                else:
                    # Select best pose in NMS-masked score volume
                    uvr_max = argmax_xyr(pscores).to(pscores)
                    pmax_score = pscores.flatten(-3).max(-1).values
                    ij_max = torch.flip(uvr_max[..., :2], dims=[-1])
                    yaw_max = 180 - uvr_max[..., 2][..., None]
                    pmap_T_maxcam = (
                        Transform2D.from_degrees(yaw_max, ij_max).float().squeeze(0)
                    )
                    ptile_T_maxcam = (
                        Transform2D.from_pixels(pmap_T_maxcam, 1 / p_ppm)
                        .float()
                        .unsqueeze(0)
                    )

                world_T_maxcam = (
                    world_T_ptile.to(ptile_T_maxcam.device) @ ptile_T_maxcam
                ).squeeze(0)

                # Prepare coarse raster bbox
                ctile_manager = data["tilers"][c_z]  # use index 0 for the 32m ppm
                min_ = np.maximum(
                    world_T_maxcam.t.cpu().numpy() - csm_coarse, world_T_ptile.t
                )
                min_ = np.maximum(min_, ctile_manager.bbox.min_)
                min_ = np.minimum(
                    min_, data_prior["canvas"][p_z].bbox.max_ - csm_coarse * 2
                )
                min_ = np.minimum(min_, ctile_manager.bbox.max_ - (csm_coarse + 1) * 2)
                max_ = min_ + 2 * csm_coarse
                bbox_ctile = BoundaryBox(min_, max_)

                # Query coarse raster
                ccanvas = ctile_manager.query(bbox_ctile)
                craster = torch.from_numpy(np.ascontiguousarray(ccanvas.raster)).long()
                craster = torch.rot90(craster, -1, dims=(-2, -1))

                # Assign queried coarse raster to the batch for forward pass
                craster = craster.unsqueeze(0).to(self.device)
                data_coarse["semantic_map"][c_z] = craster

                pmap_min = Transform2D.to_pixels(
                    world_T_ptile.inv() @ bbox_ctile.min_, 1 / p_ppm
                ).squeeze()

                # plot the topk tiles
                p_topk_coords.append((pmap_min, pmap_crad * 2, pmap_crad * 2))

                # mask scores - NMS
                x_start, y_start = (pmap_min + 0.25 * pmap_crad).to(int).numpy()
                x_end, y_end = (pmap_min + 1.75 * pmap_crad).to(int).numpy()
                pscores[..., x_start:x_end, y_start:y_end, :] = pred_prior[p_z][
                    "scores"
                ].min()
                x_start, y_start = (
                    (pmap_T_maxcam.t - 0.75 * pmap_crad).cpu().to(int).numpy()
                )
                x_end, y_end = (
                    (pmap_T_maxcam.t + 0.75 * pmap_crad).cpu().to(int).numpy()
                )
                pscores[..., x_start:x_end, y_start:y_end, :] = pred_prior[p_z][
                    "scores"
                ].min()

                world_T_ctile = Transform2D.from_Rt(
                    torch.eye(2), bbox_ctile.min_
                ).float()
                ctile_T_ptile = world_T_ctile.inv() @ world_T_ptile
                # ctile_T_cam = (
                #     ctile_T_ptile.to(model.device) @ data_prior["tile_T_cam"][p_z]
                # )
                # cmap_T_cam = Transform2D.to_pixels(ctile_T_cam, 1 / ccanvas.ppm)

                # TODO: mask edges of the prior scores?

                # logger.info(f"Arrived!")
                # exit()

            # disable fine branch of network
            with read_write(self.model.bev_mapper.conf):
                self.model.bev_mapper.conf.z_max[0] = None
                self.model.bev_mapper.conf.z_max[1] = c_z

            # Coarse Localization
            self.model.bev_cache = bev_cache_coarse
            pred_coarse = self.model(data_coarse)
            bev_cache_coarse = self.model.bev_cache
            cscores = pred_coarse[c_z]["scores"].clone()

            c_topk_coords = []

            # Loop through the topk fine maps
            for k_idx_fine in range(num_k_fine):
                if k_idx_fine == 0:
                    cmap_T_maxcam = pred_coarse[c_z]["map_T_cam_max"].float().squeeze(0)
                    ctile_T_maxcam = pred_coarse[c_z]["tile_T_cam_max"].float()
                else:
                    uvr_max = argmax_xyr(cscores).to(cscores)
                    cmax_score = cscores.flatten(-3).max(-1).values
                    ij_max = torch.flip(uvr_max[..., :2], dims=[-1])
                    yaw_max = 180 - uvr_max[..., 2][..., None]
                    cmap_T_maxcam = (
                        Transform2D.from_degrees(yaw_max, ij_max).float().squeeze(0)
                    )
                    ctile_T_maxcam = (
                        Transform2D.from_pixels(cmap_T_maxcam, 1 / c_ppm)
                        .float()
                        .unsqueeze(0)
                    )
                world_T_maxcam = (
                    world_T_ctile.to(ctile_T_maxcam.device) @ ctile_T_maxcam
                ).squeeze(0)

                # Prepare fine bbox
                ftile_manager = data["tilers"][f_z]
                min_ = np.maximum(
                    world_T_maxcam.t.cpu().numpy() - csm_fine, world_T_ctile.t
                )
                min_ = np.maximum(min_, ftile_manager.bbox.min_)
                min_ = np.minimum(min_, world_T_ctile.t + csm_coarse * 2 - csm_fine * 2)
                min_ = np.minimum(min_, ftile_manager.bbox.max_ - (csm_fine + 0.5) * 2)
                max_ = min_ + 2 * csm_fine
                bbox_ftile = BoundaryBox(min_, max_)

                # Query fine raster
                fcanvas = ftile_manager.query(bbox_ftile)
                fraster = torch.from_numpy(np.ascontiguousarray(fcanvas.raster)).long()
                fraster = torch.rot90(fraster, -1, dims=(-2, -1))

                data_fine = disable_key(data, [c_z])
                data_fine = disable_key(data, [p_z], remove_key=True)
                data_fine["semantic_map"][f_z] = fraster.unsqueeze(0).to(self.device)

                cmap_min = Transform2D.to_pixels(
                    world_T_ctile.inv() @ bbox_ftile.min_, 1 / c_ppm
                ).squeeze()

                # plot topk fine maps
                c_topk_coords.append((cmap_min, cmap_frad * 2, cmap_frad * 2))

                # mask scores - NMS
                # x_start, y_start = (cmap_min + 0.25*cmap_frad).to(int).numpy()
                # x_end, y_end = (cmap_min + 1.75*cmap_frad).to(int).numpy()
                x_start, y_start = (
                    (cmap_T_maxcam.t - 0.75 * cmap_frad).cpu().to(int).numpy()
                )
                x_end, y_end = (
                    (cmap_T_maxcam.t + 0.75 * cmap_frad).cpu().to(int).numpy()
                )
                cscores[..., x_start:x_end, y_start:y_end, :] = pred_coarse[c_z][
                    "scores"
                ].min()

                # disable coarse branch for forward pass
                with read_write(self.model.bev_mapper.conf):
                    self.model.bev_mapper.conf.z_max[0] = f_z
                    self.model.bev_mapper.conf.z_max[1] = None

                # Forward pass through fine branch
                self.model.bev_cache = bev_cache_fine
                pred_fine = self.model(data_fine)
                bev_cache_fine = self.model.bev_cache
                fscores = pred_fine[f_z]["scores"]

                # Sample coarse scores for all fine points.
                width = depth = csm_fine * 2
                cell_size = 1 / f_ppm
                grid = grids.Grid2D.from_extent_meters((width, depth), cell_size)
                ftile_xy_pts = grid.index_to_xyz(grid.grid_index())
                fmap_xy_pts = Transform2D.to_pixels(ftile_xy_pts, 1 / f_ppm)
                world_T_ftile = Transform2D.from_Rt(
                    torch.eye(2), bbox_ftile.min_
                ).float()
                ctile_T_ftile = world_T_ctile.inv() @ world_T_ftile
                ctile_xy_pts = ctile_T_ftile @ ftile_xy_pts
                cmap_xy_pts = Transform2D.to_pixels(ctile_xy_pts, 1 / c_ppm)
                cmap_interp, _, _ = grids.interpolate_nd(
                    pred_coarse[c_z]["scores"].moveaxis(-1, -3).squeeze(0),
                    cmap_xy_pts.to(cscores.device).reshape(-1, 2),  # 8, H, W  # I*J, 2
                )

                # CHECKPOINT FOR DEBUG
                cmap_interp = (
                    cmap_interp.unsqueeze(0)
                    .moveaxis(-2, -1)
                    .reshape(1, 256, 256, 64)  # TODO: use variables
                )

                # Chain probabilities of the fine and coarse
                log_probs = [
                    log_softmax_spatial(fscores),
                    log_softmax_spatial(cmap_interp),
                ]
                log_probs_chained = log_softmax_spatial(torch.stack(log_probs).sum(0))

                # Extract joint best pose
                uvr_max = argmax_xyr(log_probs_chained)
                ij_max = torch.flip(uvr_max[..., :2], dims=[-1])
                yaw_max = 180 - uvr_max[..., -1]
                fmap_T_maxcam = Transform2D.from_degrees(yaw_max.unsqueeze(-1), ij_max)
                ftile_T_maxcam = Transform2D.from_pixels(fmap_T_maxcam, 1 / f_ppm)

                max_score = (fscores + cmap_interp).flatten(-3).max(-1).values

                # Store predicted pose and score.
                topk_poses_fine.append(
                    (
                        {
                            c_z: deepcopy(pred_coarse[c_z]),
                            f_z: deepcopy(pred_fine[f_z]),
                            "chain": {
                                "tile_T_cam_max": ftile_T_maxcam,
                                "map_T_cam_max": fmap_T_maxcam,
                                "max_score": max_score,
                                "log_probs": log_probs_chained,
                            },
                        },
                        fcanvas,
                        fraster,  # .unsqueeze(0).to(self.device),
                        bbox_ftile,
                        ccanvas,
                        craster,
                        bbox_ctile,
                        c_topk_coords,
                    )
                )
            # Reset the cache for the next forward pass
            bev_cache_fine = None

        bev_cache_coarse = None

        # Select best pose out of topk
        (
            best_pred,
            fcanvas,
            fraster,
            bbox_ftile,
            ccanvas,
            craster,
            bbox_ctile,
            c_topk_coords,
        ) = max(topk_poses_fine, key=lambda x: x[0]["chain"]["max_score"])

        pred = {}
        pred["features_image"] = pred_coarse["features_image"]
        pred[c_z] = best_pred[c_z]
        pred[f_z] = best_pred[f_z]
        pred["chain"] = best_pred["chain"]

        if use_prior:
            pred_prior[p_z]["topk"] = p_topk_coords

        pred[c_z]["topk"] = c_topk_coords

        if use_prior:
            data["semantic_map"][c_z] = craster

        data["semantic_map"][f_z] = data["semantic_map"]["chain"] = fraster
        pred["chain"]["features_map"] = pred[f_z]["features_map"]

        # logger.info("Arrived!!")
        # exit()

        # with torch.no_grad():
        #     pred = self.model(data_)

        # xy_gps = canvas[z_max_list[0]].bbox.center
        # uv_gps = torch.from_numpy(canvas.to_uv(xy_gps))

        # lp_xyr = pred["log_probs"].squeeze(0)
        # tile_size = canvas.bbox.size.min() / 2
        # sigma = tile_size - 20  # 20 meters margin
        # lp_xyr = fuse_gps(
        #     lp_xyr,
        #     uv_gps.to(lp_xyr),
        #     self.config.model.pixel_per_meter,
        #     sigma=sigma,
        # )
        # xyr = argmax_xyr(lp_xyr).cpu()

        # Chain log_probs of 32m and 128m branches
        # scores = [pred[k]["scores"].to("cpu") for k in self.config.data.z_max]

        # fine_num_pixels = max([score_volume.shape[-2] for score_volume in scores])
        # upsample_ppm = max(self.config.data.pixel_per_meter)
        # h = w = fine_num_pixels
        # scores = [
        #     torch.nn.functional.interpolate(
        #         score.moveaxis(-1, -3), size=(int(h), int(w)), mode="bilinear"
        #     ).moveaxis(-3, -1)
        #     for score in scores
        # ]
        # log_probs = [log_softmax_spatial(score) for score in scores]
        # log_probs_chained = log_softmax_spatial(torch.stack(log_probs).sum(0))
        # probs_chained = log_probs_chained.exp().cpu()
        # del scores, log_probs

        # uvr_max = argmax_xyr(log_probs_chained)
        # ij_max = torch.flip(uvr_max[..., :2], dims=[-1])
        # yaw_max = 180 - uvr_max[..., -1]
        # map_T_max = Transform2D.from_degrees(yaw_max.unsqueeze(-1), ij_max)

        # tile_T_cam_max_chained = (
        #     Transform2D.from_pixels(map_T_max, 1 / upsample_ppm)
        # ).cpu()

        if use_prior:
            data["semantic_map"][p_z] = data_prior["semantic_map"][p_z]
            pred[p_z] = pred_prior[p_z]

        return (data, pred)

    def run(
        self,
        image_path="assets/query_vancouver_1.jpeg",
        prior_address="Vancouver Waterfront Station",
        tile_size_meters=128,
        out_dir=None,
        hierarchical=False,
        topk=None,
    ):

        if hierarchical:
            assert isinstance(topk, int) and 0 < topk <= 10

        # Read+calibrate image, create bbox from address
        image, camera, roll_pitch, proj, bbox = preprocess_inputs(
            image_path,
            prior_address=prior_address,
            tile_size_meters=tile_size_meters,  # try 64, 256, etc.
        )
        data = self.prepare_inputs(
            image, camera, proj, bbox, hierarchical, roll_pitch=roll_pitch
        )
        logger.info("Finished preparing model inputs. Starting inference...")
        if hierarchical:
            data, pred = self.localize_hierarchical(data, topk)
        else:
            (
                tile_T_cam_max,
                map_T_cam_max,
                prob,
                neural_map,
                image,
                semantic_map,
                pred,
            ) = self.localize(
                data,
            )

        logger.info("Inference complete. Preparing visualizations...")
        plot_results(data, pred, out_dir, image_path)
        logger.info(f"Visualizations saved to {out_dir}")
        return


def plot_results(data, pred, out_dir=None, image_path=None):

    keys = [key for key in pred.keys() if isinstance(key, float)]
    print(f"Keys: {keys}")
    keys += ["chain"] if "chain" in pred.keys() else []

    for index, k in enumerate(keys):

        lp_ijt = pred[k]["log_probs"]
        assert lp_ijt.device != "cuda"
        prob = lp_ijt.exp()
        # depth32m = self.config.data.z_max[0]  # corresponds to finer ppm
        # Get the memory-layout map raster from localize() output
        map_viz = Colormap.apply(data["semantic_map"][k].squeeze(0))
        if tuple(prob.shape) != tuple(map_viz.shape[:2]):
            map_viz = (
                torch.nn.functional.interpolate(
                    torch.from_numpy(map_viz).moveaxis(-1, -3).unsqueeze(0),
                    size=tuple(prob.shape[-3:-1]),
                )
                .squeeze(0)
                .moveaxis(-3, -1)
                .numpy()
            )
        (neural_map_rgb,) = features_to_RGB(pred[k]["features_map"].squeeze(0).numpy())
        overlay = likelihood_overlay(
            prob.squeeze(0).numpy().max(-1), map_viz.mean(-1, keepdims=True)
        )
        overlay, neural_map_rgb, map_viz = [
            np.swapaxes(x, 0, 1) for x in (overlay, neural_map_rgb, map_viz)
        ]
        plot_images(
            [
                data["image"].squeeze(0).permute(1, 2, 0),
                map_viz,
                overlay,
                neural_map_rgb,
            ],
            titles=["input image", "OpenStreetMap raster", "prediction", "neural map"],
            origins=["upper", "lower", "lower", "lower"],
        )
        axes = plt.gcf().axes
        ax = axes[2]
        # ax.scatter(*data['canvas'][32.0].to_uv(bbox.center), s=5, c="red")
        plot_dense_rotations(ax, prob.squeeze(0), w=0.005, s=1 / 25)
        if k in [32.0, "chain"]:
            side = map_viz.shape[0]
            plot_pose(
                [1],
                pred[k]["map_T_cam_max"].t.squeeze(0),
                pred[k]["map_T_cam_max"].angle.squeeze(0),
                c="k",
                refactored=True,
                dot=False,
                s=side / 256,
            )

        if out_dir is None:
            plt.show()
        else:
            Path(out_dir).mkdir(exist_ok=True, parents=True)
            p = str(Path(out_dir) / Path(image_path).stem) + f"_{k}_{{}}.png"
            save_plot(p.format("pred"))
            plt.close()

    return


if __name__ == "__main__":
    model = OrienterNetv2(
        experiment_or_path="OrienterNetv2",
        num_rotations=32,
        # device="cpu",
    )
    model.run(
        image_path="assets/query_vancouver_3.jpeg",
        prior_address="Vancouver Waterfront Station",
        tile_size_meters=275,
        out_dir="demo_figures/",
        hierarchical=True,
        topk=1,
    )


# TODO List:
# add topk visualization
