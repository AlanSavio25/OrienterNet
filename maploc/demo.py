# Copyright (c) Meta Platforms, Inc. and affiliates.
from maploc.osm.tiling import TileManager
from maploc.utils.viz_localization import (
    likelihood_overlay,
    plot_dense_rotations,
    add_circle_inset,
)
from maploc.utils.viz_2d import features_to_RGB
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images

from typing import Optional, Tuple, Dict

from maploc.utils.viz_localization import plot_pose, plot_bev
from maploc.utils.wrappers import Transform2D
import numpy as np
import torch
from pathlib import Path

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
from scipy.spatial.transform import Rotation

from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_utilities.core.apply_func import apply_to_collection
import matplotlib.pyplot as plt
from .utils.viz_2d import features_to_RGB, plot_images, save_plot

try:
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="orienternet")
except ImportError:
    geolocator = None

try:
    from gradio_client import Client

    calibrator = Client("https://jinlinyi-perspectivefields.hf.space/")
    # calibrator = None
except (ImportError, ValueError):
    calibrator = None


def image_calibration(image_path):
    logger.info("Calling the PerspectiveFields calibrator, this may take some time.")
    result = calibrator.predict(
        # image_path, "NEW:Paramnet-360Cities-edina-centered", api_name="/predict" # broken
        image_path,
        "PersNet_Paramnet-GSV-centered",
        api_name="/predict",
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
    if calibrator is not None:
        roll_pitch, fov = image_calibration(image_path)
        logger.info("Using (roll, pitch) %s.", roll_pitch)
    else:
        logger.info("Could not call PerspectiveFields, maybe install gradio_client?")

    # logger.info("Using cam_R_gcam %s.", cam_R_gcam)

    camera = camera_from_exif(exif, fov)
    if camera is None:
        raise ValueError(
            "No camera intrinsics found in the EXIF, provide an FoV guess."
        )

    proj = Projection(*latlon)
    center = proj.project(latlon)
    bbox = BoundaryBox(center, center) + tile_size_meters
    return image, camera, roll_pitch, proj, bbox, latlon


class Demo:
    def __init__(
        self,
        load_from_hub=False,
        experiment_or_path: Optional[str] = "OrienterNet_MGL",
        device=None,
        **kwargs,
    ):

        if load_from_hub:
            CHECKPOINT_URL = "https://github.com/AlanSavio25/OrienterNet/releases/download/prerelease/prerelease.ckpt"
            ckpt_path = Path("./experiment_demo") / experiment_or_path
            if not ckpt_path.exists():
                ckpt_path.parent.mkdir(exist_ok=True, parents=True)
                torch.hub.download_url_to_file(CHECKPOINT_URL, ckpt_path)

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
        # cam_R_gcam: torch.Tensor
        roll_pitch: Optional[Tuple[float]] = None,
    ):
        assert image.shape[:2][::-1] == tuple(camera.size.tolist())
        target_focal_length = self.config.data.resize_image / 2
        factor = target_focal_length / camera.f
        size = (camera.size * factor).round().int()

        image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
        valid = None
        # we don't need to rectify if we're using 3D grid projection. cam_R_gcam must be provided to model.
        if roll_pitch is not None:
            roll, pitch = roll_pitch
            R = Rotation.from_euler("ZX", (-roll, -pitch), degrees=True).as_matrix()
            R = torch.from_numpy(R)
            image, valid = rectify_image(image, camera.float(), cam_R_gcam=R)
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
            roll_pitch=roll_pitch,
            cam_R_gcam=torch.eye(3),
            camera=camera.float(),
            valid=valid,
        )

    def localize(self, image: np.ndarray, camera: Camera, canvas: Canvas, **kwargs):
        data = self.prepare_data(image, camera, canvas, **kwargs)
        data_ = apply_to_collection(data, (torch.Tensor, Camera), lambda x: x[None])
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

    def run_demo(
        self,
        image_path="assets/query_vancouver_1.jpeg",
        prior_address="Vancouver Waterfront Station",
        out_dir=None,
    ):

        image, camera, roll_pitch, proj, bbox, prior_latlon = read_input_image(
            image_path,
            prior_address=prior_address,
            tile_size_meters=128,  # try 64, 256, etc.
        )
        logger.info(f"Finished reading and calibrating image")

        ppm_list = self.config.data.pixel_per_meter
        canvas = {}
        for i, z_max in enumerate(self.config.data.z_max):
            ppm = ppm_list[i]
            tiler = TileManager.from_bbox(proj, bbox + 10, ppm)
            canvas[z_max] = tiler.query(bbox)

        # Show the inputs to the model: image and raster map
        # from maploc.osm.viz import Colormap, plot_nodes
        # from maploc.utils.viz_2d import plot_images

        # depth32m = self.config.data.z_max[0]  # corresponds to finer ppm
        # map_viz = Colormap.apply(canvas[depth32m].raster)
        # plot_images([image, map_viz], titles=["input image", "OpenStreetMap raster"])
        # plot_nodes(1, canvas[depth32m].raster[2], fontsize=6, size=10)

        logger.info("Finished pulling OSM data. Starting inference...")
        # Run the inference
        tile_T_cam_max, map_T_cam_max, prob, neural_map, image, semantic_map, pred = (
            self.localize(
                image,
                camera,
                canvas,
                roll_pitch=roll_pitch,  # cam_R_gcam=cam_R_gcam
            )
        )

        logger.info("Inference complete. Preparing visualizations...")

        depth32m = self.config.data.z_max[0]  # corresponds to finer ppm
        # Get the memory-layout map raster from localize() output
        map_viz = Colormap.apply(semantic_map[depth32m])

        # Visualize the predictions
        overlay = likelihood_overlay(
            prob.squeeze(0).numpy().max(-1), map_viz.mean(-1, keepdims=True)
        )
        (neural_map_rgb,) = features_to_RGB(neural_map.squeeze(0).numpy())

        overlay, neural_map_rgb, map_viz = [
            np.swapaxes(x, 0, 1) for x in (overlay, neural_map_rgb, map_viz)
        ]

        map_T_cam_max = Transform2D.to_pixels(tile_T_cam_max, 1 / 2)

        plot_images(
            [image.permute(1, 2, 0), map_viz, overlay, neural_map_rgb],
            titles=["input image", "OpenStreetMap raster", "prediction", "neural map"],
            origins=["upper", "lower", "lower", "lower"],
        )
        axes = plt.gcf().axes
        ax = axes[2]
        ax.scatter(*canvas[32.0].to_uv(bbox.center), s=5, c="red")
        plot_dense_rotations(ax, prob.squeeze(0), w=0.005, s=1 / 25)

        plot_pose(
            [1],
            map_T_cam_max.t.squeeze(0),
            map_T_cam_max.angle.squeeze(0),
            c="k",
            refactored=True,
            dot=False,
            s=1 / 60,
        )
        # add_circle_inset(ax, uv) # still broken

        # BEV overlay. broken
        # (bev,) = features_to_RGB(pred[128.0]["features_bev"].squeeze(0).numpy(), masks=[pred[128.0]["valid_bev"].squeeze(0).numpy()])
        # increase 128m bev's resolution
        # bev = torch.nn.functional.interpolate(
        #                 torch.from_numpy(bev).unsqueeze(0).moveaxis(-1, -3), scale_factor=4, mode="bilinear"
        #             ).moveaxis(-3, -1).squeeze(0).numpy()
        # axes[1].images[0].set_interpolation("none")
        # plot_bev(bev, uv=map_T_cam_max.t.squeeze(0), yaw=map_T_cam_max.angle.squeeze(0), zorder=10, ax=axes[2])

        if out_dir is None:
            plt.show()
        else:
            p = str(Path(out_dir) / Path(image_path).stem) + f"_{{}}.png"
            save_plot(p.format("pred"))
            plt.close()

        # Visualize BEV + Confidence
        titles = []
        bevs = []
        for z_max in self.config.data.z_max:
            mask_bev = pred[z_max]["valid_bev"].squeeze(0)
            (bev,) = features_to_RGB(
                pred[z_max]["features_bev"].squeeze(0).numpy(), masks=[mask_bev.numpy()]
            )
            conf_q = torch.log(pred[z_max]["bev"]["confidence"])
            conf_q = conf_q.masked_fill(~mask_bev, np.nan)
            bevs.append(np.swapaxes(bev, 0, 1))
            bevs.append(np.swapaxes(conf_q.squeeze(0), 0, 1))
            titles.append(f"BEV depth: {z_max}m")
            titles.append(f"BEV Confidence: {z_max}m")
        plot_images(bevs, titles, origins=["lower", "lower"] * 2, cmaps="jet")
        if out_dir is None:
            plt.show()
        else:
            save_plot(p.format("bev"))
            plt.close()

        # Visualize Image Features and Max Scale Score
        (fine_im_feat,) = features_to_RGB(pred["features_image"][0, :128, ...].numpy())
        (coarse_im_feat,) = features_to_RGB(
            pred["features_image"][0, 128:, ...].numpy()
        )

        max_scores = []
        for z_max in self.config.data.z_max:
            scales_scores = pred[z_max]["pixel_scales"].squeeze(0)
            log_prob = torch.nn.functional.log_softmax(scales_scores, dim=-1)
            max_scores.append(log_prob.max(-1).values.exp())

        plot_images(
            [fine_im_feat, coarse_im_feat, *max_scores],
            titles=[
                "Fine Image Features",
                "Coarse Image Features",
                "Fine Max Scale Score (red=high)",
                "Coarse Max Scale Score (red=high)",
            ],
            origins=["upper", "upper", "upper", "upper"],
            cmaps="jet",
        )

        if out_dir is None:
            plt.show()
        else:
            save_plot(p.format("scales"))
            plt.close()

        logger.info(f"Done")

        return


if __name__ == "__main__":
    demo = Demo(
        load_from_hub=True,
        experiment_or_path="prerelease/prerelease.ckpt",
        num_rotations=64,
        device="cpu",
    )
    demo.run_demo(
        image_path="assets/query_vancouver_1.jpeg",
        prior_address="Vancouver Waterfront Station",
        out_dir="demo_figures/",
    )
