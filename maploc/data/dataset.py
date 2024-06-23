# Copyright (c) Meta Platforms, Inc. and affiliates.

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as tvf
from omegaconf import DictConfig, OmegaConf

from ..models.utils import rotmat2d
from ..osm.tiling import TileManager
from ..utils.geo import BoundaryBox
from ..utils.io import read_image
from ..utils.wrappers import Camera, Transform2D, Transform3D
from .image import pad_image, rectify_image, resize_image
from .utils import compose_rotmat, decompose_cam_into_gcam, random_flip, random_rot90


class MapLocDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "accuracy_gps": 15,
        "random": True,
        "num_threads": None,
        # map
        "return_multiscale": False,
        "z_max": 32.0,
        "bev_ppm": 2,
        "num_classes": None,
        "pixel_per_meter": "???",
        "crop_size_meters": "???",
        "max_init_error": "???",
        "max_init_error_rotation": None,
        "init_from_gps": False,
        "return_gps": False,
        "force_camera_height": None,
        # pose priors
        "add_map_mask": False,
        "mask_radius": None,
        "mask_pad": 1,
        "prior_range_rotation": None,
        # image preprocessing
        "target_focal_length": None,
        "reduce_fov": None,
        "resize_image": None,
        "pad_to_square": False,  # legacy
        "pad_to_multiple": 32,
        "rectify_image": True,  # set to False when BEV inference is inverse.
        "augmentation": {
            "rot90": False,
            "flip": False,
            "image": {
                "apply": False,
                "brightness": 0.5,
                "contrast": 0.4,
                "saturation": 0.4,
                "hue": 0.5 / 3.14,
            },
        },
    }

    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        names: List[str],
        data: Dict[str, Any],
        image_dirs: Dict[str, Path],
        tile_managers: Dict[str, TileManager],
        image_ext: str = "",
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.data = data
        self.image_dirs = image_dirs
        self.tile_managers = tile_managers
        self.names = names
        self.image_ext = image_ext

        tfs = []
        if stage == "train" and cfg.augmentation.image.apply:
            args = OmegaConf.masked_copy(
                cfg.augmentation.image, ["brightness", "contrast", "saturation", "hue"]
            )
            tfs.append(tvf.ColorJitter(**args))
        self.tfs = tvf.Compose(tfs)

        if not isinstance(list(self.tile_managers.values())[0], list):
            for scene in self.tile_managers:
                self.tile_managers[scene] = [self.tile_managers[scene]]*len(self.cfg.crop_size_meters)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)

        scene, seq, name = self.names[idx]
        if self.cfg.init_from_gps:
            latlon_gps = self.data["gps_position"][idx][:2].clone().numpy()
            xy_w_init = self.tile_managers[scene].projection.project(latlon_gps)
        else:
            xy_w_init = self.data["t_c2w"][idx][:2].clone().double().numpy()

        if "shifts" in self.data:
            yaw = self.data["roll_pitch_yaw"][idx][-1]
            world_R_cam = rotmat2d(np.deg2rad((90 - yaw))).float()
            error = (world_R_cam @ self.data["shifts"][idx][:2]).numpy()
        else:
            error = np.random.RandomState(seed).uniform(-1, 1, size=2)

        if self.cfg.return_multiscale:
            xy_w_init += error * min(self.cfg.max_init_error)
            bbox_tile = [
                BoundaryBox(xy_w_init - crop_size_meters, xy_w_init + crop_size_meters)
                for crop_size_meters in self.cfg.crop_size_meters
            ]
        else:
            xy_w_init += error * self.cfg.max_init_error
            bbox_tile = BoundaryBox(
                xy_w_init - self.cfg.crop_size_meters,
                xy_w_init + self.cfg.crop_size_meters,
            )
        return self.get_view(idx, scene, seq, name, seed, bbox_tile)

    def get_view(self, idx, scene, seq, name, seed, bbox_tile):
        data = {
            "index": idx,
            "name": name,
            "scene": scene,
            "sequence": seq,
        }
        cam_dict = self.data["cameras"][scene][seq][self.data["camera_id"][idx]]
        cam = Camera.from_dict(cam_dict).float()

        # for backward compatibility
        if "roll_pitch_yaw" in self.data:
            world_R_cam = compose_rotmat(*self.data["roll_pitch_yaw"][idx].numpy())
        else:
            world_R_cam = self.data["R_c2w"][idx].numpy()

        world_t_cam = self.data["t_c2w"][idx].numpy()

        image = read_image(self.image_dirs[scene] / (name + self.image_ext))
        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )

        if self.cfg.force_camera_height is not None:
            data["camera_height"] = torch.tensor(self.cfg.force_camera_height)
        elif "camera_height" in self.data:
            data["camera_height"] = self.data["height"][idx].clone()

        # raster extraction
        if self.cfg.return_multiscale:
            z_max = self.cfg.z_max
            canvas = [
                tile_manager.query(bbox_tile)
                for (tile_manager, bbox_tile) in zip(
                    self.tile_managers[scene], bbox_tile
                )
            ]
            ppm = {z: torch.tensor(c.ppm).float() for z, c in zip(z_max, canvas)}
            # raster = [c.raster for c in canvas]
            # raster = {torch.from_numpy(np.ascontiguousarray(r)).long() for r in raster]
            raster = {
                z: torch.from_numpy(np.ascontiguousarray(c.raster)).long()
                for z, c in zip(z_max, canvas)
            }
            # TODO: dict aerials
            if hasattr(canvas[0], "aerial"):
                aerial = [c.aerial for c in canvas]
                aerial = torch.stack(
                    [
                        torch.from_numpy(np.ascontiguousarray(aerial_)).long()
                        for aerial_ in aerial
                    ]
                )
        else:
            canvas = self.tile_managers[scene].query(bbox_tile)
            ppm = torch.tensor(canvas.ppm).float()
            raster = canvas.raster  # C, H, W
            raster = torch.from_numpy(np.ascontiguousarray(raster)).long()
            if hasattr(canvas, "aerial"):
                aerial = canvas.aerial
                aerial = torch.from_numpy(np.ascontiguousarray(aerial)).long()

        world_T_cam = Transform3D.from_Rt(world_R_cam, world_t_cam)
        world_T_cam2d = Transform2D.camera_2d_from_3d(world_T_cam)

        _, cam_R_gcam = decompose_cam_into_gcam(world_T_cam)

        if self.cfg.return_multiscale:
            world_T_tile = {
                z: Transform2D.from_Rt(torch.eye(2), c.bbox.min_) for z, c in zip(z_max, canvas)
            }
            tile_T_cam = {
                z: (w_T_t.inv() @ world_T_cam2d).float()
                for z, w_T_t in world_T_tile.items()
            }
        else:
            world_T_tile = Transform2D.from_Rt(torch.eye(2), canvas.bbox.min_)
            tile_T_cam = (world_T_tile.inv() @ world_T_cam2d).float()

        # Map augmentations
        if self.stage == "train" and not self.cfg.return_multiscale:
            if self.cfg.augmentation.rot90:
                raster, tile_T_cam = random_rot90(raster, tile_T_cam, canvas.ppm)
            if self.cfg.augmentation.flip:
                image, raster, tile_T_cam, cam_R_gcam = random_flip(
                    image, raster, tile_T_cam, cam_R_gcam, canvas.ppm
                )
        if self.cfg.return_multiscale:
            map_T_cam = {
                z: Transform2D.to_pixels(t_T_c, 1 / c.ppm)
                for (z, t_T_c, c) in zip(z_max, tile_T_cam.values(), canvas)
            }
        else:
            map_T_cam = Transform2D.to_pixels(tile_T_cam, 1 / canvas.ppm)
        # map_T_cam will be deprecated, tile_T_cam is sufficient.

        # We can avoid rectification when using SNAP's inverse BEV prediction
        image, valid, cam = self.process_image(
            image, cam, seed, cam_R_gcam, rectify=self.cfg.rectify_image
        )
        if self.cfg.rectify_image:
            cam_R_gcam = torch.eye(3)

        # Spatial to memory layout
        if self.cfg.return_multiscale:
            # raster = {k:torch.rot90(r, -1, dims=(-2, -1)) for k,r in raster.items()}
            for k, r in raster.items():
                raster[k] = torch.rot90(r, -1, dims=(-2, -1))
            if hasattr(canvas, "aerial"):
                aerial = torch.stack(
                    [torch.rot90(a, -1, dims=(-2, -1)) for a in aerial]
                )
                data["aerial_map"] = aerial

            world_t_init = {
                z: torch.from_numpy(bbox_tile_.center)
                for (z, bbox_tile_) in zip(z_max, bbox_tile)
            }
            tile_t_init = {
                z: (w_t_init - w_T_t.t).float()
                for (z, w_t_init, w_T_t) in zip(
                    z_max, world_t_init.values(), world_T_tile.values()
                )
            }
            map_t_init = {
                z: Transform2D.to_pixels(t_t_init, 1 / c.ppm)
                for (z, t_t_init, c) in zip(z_max, tile_t_init.values(), canvas)
            }
        else:
            raster = torch.rot90(raster, -1, dims=(-2, -1))
            if hasattr(canvas, "aerial"):
                aerial = torch.rot90(aerial, -1, dims=(-2, -1))
                data["aerial_map"] = aerial

            world_t_init = torch.from_numpy(bbox_tile.center)
            tile_t_init = (world_t_init - world_T_tile.t).float()
            map_t_init = Transform2D.to_pixels(tile_t_init, 1 / canvas.ppm)

        # Create the mask for prior location
        if self.cfg.add_map_mask:
            if self.cfg.return_multiscale:
                map_mask = {
                    z: torch.from_numpy(self.create_map_mask(c, init_error, pad))
                    for (z, c, init_error, pad) in zip(
                        z_max, canvas, self.cfg.max_init_error, self.cfg.mask_pad
                    )
                }
                data["map_mask"] = {
                    z: torch.rot90(mask, -1, dims=(-2, -1))
                    for (z, mask) in zip(z_max, map_mask.values())
                }
            else:
                map_mask = torch.from_numpy(
                    self.create_map_mask(
                        canvas, self.cfg.max_init_error, self.cfg.mask_pad
                    )
                )
                data["map_mask"] = torch.rot90(map_mask, -1, dims=(-2, -1))

        if (
            self.cfg.max_init_error_rotation is not None
        ):  # does not support multiscale yet
            if "shifts" in self.data:
                error = self.data["shifts"][idx][-1]
            else:
                error = np.random.RandomState(seed + 1).uniform(-1, 1)
                error = torch.tensor(error, dtype=torch.float)
            yaw_init = tile_T_cam.angle + error * self.cfg.max_init_error_rotation
            range_ = self.cfg.prior_range_rotation or self.cfg.max_init_error_rotation
            data["yaw_prior"] = torch.stack([yaw_init, torch.tensor(range_)])

        if self.cfg.return_gps:
            gps = self.data["gps_position"][idx][:2].numpy()
            if self.cfg.return_multiscale:
                world_t_gps = self.tile_managers[scene][0].projection.project(gps)
                world_t_gps = torch.from_numpy(world_t_gps)
                tile_t_gps = [
                    (world_t_gps - w_T_t.t).float() for w_T_t in world_T_tile.values()
                ]
                data["tile_t_gps"] = {
                    z: t_t_gps
                    for (z, t_t_gps) in zip(z_max, tile_t_gps)
                }
                data["map_t_gps"] = {
                    z: Transform2D.to_pixels(t_t_gps, 1 / c.ppm)
                    for (z, t_t_gps, c) in zip(z_max, tile_t_gps, canvas)
                }
                data["accuracy_gps"] = {
                    z: torch.tensor(min(self.cfg.accuracy_gps, crop_size_meters))
                    for (z, crop_size_meters) in zip(z_max, self.cfg.crop_size_meters)
                }

            else:
                world_t_gps = self.tile_managers[scene].projection.project(gps)
                world_t_gps = torch.from_numpy(world_t_gps)
                tile_t_gps = (world_t_gps - world_T_tile.t).float()
                data["tile_t_gps"] = tile_t_gps
                data["map_t_gps"] = Transform2D.to_pixels(tile_t_gps, 1 / canvas.ppm)
                data["accuracy_gps"] = torch.tensor(
                    min(self.cfg.accuracy_gps, self.cfg.crop_size_meters)
                )

        if "chunk_index" in self.data:
            data["chunk_id"] = (scene, seq, self.data["chunk_index"][idx])

        if self.cfg.return_multiscale:
            canvas = {z: c for z,c in zip(z_max, canvas)}
            z_max = {z: torch.tensor([z]).float() for z in z_max}
            bev_ppm = {z: torch.tensor([bev_ppm]).float() for (z, bev_ppm) in zip(z_max,self.cfg.bev_ppm)}

        return {
            **data,
            # Image
            "image": image,
            "valid": valid,
            "camera": cam,
            # GT Pose
            "world_T_cam": world_T_cam.t,
            "cam_R_gcam": cam_R_gcam,
            # Map(s)
            "semantic_map": raster,
            "canvas": canvas,
            "tile_T_cam": tile_T_cam,
            "map_T_cam": map_T_cam,
            "map_t_init": map_t_init,
            "pixels_per_meter": ppm,
            "z_max": z_max,
            "bev_ppm": bev_ppm,
        }

    def process_image(self, image, cam, seed, cam_R_gcam, rectify=True):

        if rectify:
            image, valid = rectify_image(image, cam, cam_R_gcam)
        else:
            valid = torch.ones(
                image.shape[:-3] + image.shape[-2:],
                dtype=torch.bool,
                device=image.device,
            )

        if self.cfg.target_focal_length is not None:
            # resize to a canonical focal length
            factor = self.cfg.target_focal_length / cam.f.numpy()
            size = (np.array(image.shape[-2:][::-1]) * factor).astype(int)
            image, _, cam, valid = resize_image(image, size, camera=cam, valid=valid)
            size_out = self.cfg.resize_image
            if size_out is None:
                # round the edges up such that they are multiple of a factor
                stride = self.cfg.pad_to_multiple
                size_out = (np.ceil((size / stride)) * stride).astype(int)
            # crop or pad such that both edges are of the given size
            image, valid, cam = pad_image(
                image, size_out, cam, valid, crop_and_center=True
            )
        elif self.cfg.resize_image is not None:
            image, _, cam, valid = resize_image(
                image, self.cfg.resize_image, fn=max, camera=cam, valid=valid
            )
            if self.cfg.pad_to_square:
                # pad such that both edges are of the given size
                image, valid, cam = pad_image(image, self.cfg.resize_image, cam, valid)

        if self.cfg.reduce_fov is not None:
            h, w = image.shape[-2:]
            f = float(cam.f[0])
            fov = np.arctan(w / f / 2)
            w_new = round(2 * f * np.tan(self.cfg.reduce_fov * fov))
            image, valid, cam = pad_image(
                image, (w_new, h), cam, valid, crop_and_center=True
            )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.tfs(image)
        return image, valid, cam

    def create_map_mask(self, canvas, max_init_error, mask_pad):
        map_mask = np.zeros(canvas.raster.shape[-2:], bool)
        radius = self.cfg.mask_radius or max_init_error
        mask_min, mask_max = np.round(
            canvas.to_uv(canvas.bbox.center)
            + np.array([[-1], [1]]) * (radius + mask_pad) * canvas.ppm
        ).astype(int)
        map_mask[mask_min[1] : mask_max[1], mask_min[0] : mask_max[0]] = True
        return map_mask
