# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional, Tuple, Dict

import numpy as np
import torch

from . import logger
from .data.image import pad_image, rectify_image, resize_image
from .evaluation.run import pretrained_models, resolve_checkpoint_path
from .models.orienternet import OrienterNet
from .models.voting import argmax_xyr, fuse_gps, log_softmax_spatial
from .osm.raster import Canvas
from .utils.exif import EXIF
from .utils.geo import BoundaryBox, Projection
from .utils.io import read_image
from .utils.wrappers import Camera, Transform2D

from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_utilities.core.apply_func import apply_to_collection

try:
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="orienternet")
except ImportError:
    geolocator = None

try:
    from gradio_client import Client

    # calibrator = Client("https://jinlinyi-perspectivefields.hf.space/")
    calibrator = None
except (ImportError, ValueError):
    calibrator = None


def image_calibration(image_path):
    logger.info("Calling the PerspectiveFields calibrator, this may take some time.")
    result = calibrator.submit(
        image_path, "NEW:Paramnet-360Cities-edina-centered", api_name="/predict"
    )
    result = dict(r.rsplit(" ", 1) for r in result[1].split("\n"))
    roll_pitch = float(result["roll"]), float(result["pitch"])
    return roll_pitch, float(result["vertical fov"])


def camera_from_exif(exif: EXIF, fov: Optional[float] = None) -> Camera:
    w, h = image_size = exif.extract_image_size()
    _, f_ratio = exif.extract_focal()
    if f_ratio == 0:
        if fov is not None:
            # This is the vertical FoV.
            f = h / 2 / np.tan(np.deg2rad(fov) / 2)
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


def read_input_image(
    image_path: str,
    prior_latlon: Optional[Tuple[float, float]] = None,
    prior_address: Optional[str] = None,
    fov: Optional[float] = None,
    tile_size_meters: int = 64,
):
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

    roll_pitch = None
    cam_R_gcam = torch.eye(3)
    calibrator = None
    if calibrator is not None:
        roll_pitch, fov = image_calibration(image_path)
        # cam_R_gcam =  # TODO: compute cam_R_gcam from roll pitch
        logger.info("Using (roll, pitch) %s.", roll_pitch)
    else:
        logger.info("Could not call PerspectiveFields, maybe install gradio_client?")


    logger.info("Using cam_R_gcam %s.", cam_R_gcam)

    camera = camera_from_exif(exif, fov)
    if camera is None:
        raise ValueError(
            "No camera intrinsics found in the EXIF, provide an FoV guess."
        )

    proj = Projection(*latlon)
    center = proj.project(latlon)
    bbox = BoundaryBox(center, center) + tile_size_meters
    return image, camera, cam_R_gcam, proj, bbox, latlon


class Demo:
    def __init__(
        self,
        experiment_or_path: Optional[str] = "OrienterNet_MGL",
        device=None,
        **kwargs
    ):
        if experiment_or_path in pretrained_models:
            experiment_or_path, _ = pretrained_models[experiment_or_path]
        path = resolve_checkpoint_path(experiment_or_path)
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

    def prepare_data(
        self,
        image: np.ndarray,
        camera: Camera,
        canvas: Dict,
        cam_R_gcam: torch.Tensor
        # roll_pitch: Optional[Tuple[float]] = None,
    ):
        assert image.shape[:2][::-1] == tuple(camera.size.tolist())
        target_focal_length = self.config.data.resize_image / 2
        factor = target_focal_length / camera.f
        size = (camera.size * factor).round().int()

        image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
        valid = None
        # we don't need to rectify if we're using 3D grid projection. cam_R_gcam must be provided to model.
        # if roll_pitch is not None:
        #     roll, pitch = roll_pitch
        #     image, valid = rectify_image(
        #         image,
        #         camera.float(),
        #         roll=-roll,
        #         pitch=-pitch,
        #     )
        image, _, camera, *maybe_valid = resize_image(
            image, size.tolist(), camera=camera, valid=valid
        )
        valid = None if valid is None else maybe_valid

        max_stride = max(self.model.bev_mapper.image_encoder.layer_strides)
        size = (torch.ceil(size / max_stride) * max_stride).int()
        image, valid, camera = pad_image(
            image, size.tolist(), camera, crop_and_center=True
        )

        semantic_map = {}
        for z_max in canvas:
            semantic_map[z_max] = torch.from_numpy(canvas[z_max].raster).long()
            semantic_map[z_max] = torch.rot90(semantic_map[z_max], -1, dims=(-2, -1))

        return dict(
            image=image,
            semantic_map=semantic_map,
            cam_R_gcam=cam_R_gcam.float(),
            camera=camera.float(),
            valid=valid,
        )

    def localize(self, image: np.ndarray, camera: Camera, canvas: Canvas, **kwargs):
        data = self.prepare_data(image, camera, canvas, **kwargs)
        data_ = apply_to_collection(data, (torch.Tensor,Camera), lambda x: x[None])
        # data_ = {k: v.to(self.device)[None] for k, v in data.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            pred = self.model(data_)

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
        scores = [pred[k]["scores"].to('cpu') for k in self.config.data.z_max]

        # crop_size_meters = model.cfg.data.crop_size_meters[0]
        fine_num_pixels = max([score_volume.shape[-2] for score_volume in scores])
        upsample_ppm = max(self.config.data.pixel_per_meter)
        h = w = fine_num_pixels 
        scores = [
            torch.nn.functional.interpolate(
                score.moveaxis(-1, -3), size=(int(h), int(w)), mode="bilinear"
            ).moveaxis(-3, -1)
            for score in scores
        ]
        log_probs = [
            log_softmax_spatial(score)
            for score in scores
        ]
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
        image = data["image"].cpu() # padded/rectified image
        semantic_map = data["semantic_map"] # this contains the memory-layout raster

        return tile_T_cam_max_chained, map_T_max, probs_chained, f_map_fine, image, semantic_map