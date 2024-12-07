import os
import torch
import numpy as np
from PIL import Image
from maploc import logger
from pathlib import Path
from pprint import pformat
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from geocalib import GeoCalib
from geocalib.gravity import Gravity
from geocalib.camera import Pinhole as Camera
from geocalib.viz2d import plot_perspective_fields, plot_confidences


from maploc.osm.tiling import TileManager
from maploc.inference import OrienterNetv2
from maploc.osm.viz import Colormap, GeoPlotter, plot_nodes
from maploc.utils.viz_2d import features_to_RGB, plot_images, save_plot
from maploc.utils.viz_localization import likelihood_overlay, plot_dense_rotations
from maploc.inference import preprocess_inputs
from maploc.utils.wrappers import Camera as Camera2, Transform2D, Transform3D
from maploc.utils.viz_localization import plot_pose

import pycolmap
from hloc import colmap_from_nvm
from sfm.database import LocalizationDatabase


class GeoRegistrationPipeline:
    """Pipeline to georegister an SfM model"""
    
    def __init__(
        self,
        prior_location="Aachen Cathedral",
        tile_size_meters=512,
        num_images=25,
        hierarchical=False,
        **kwargs
    ):

        logger.info("Initializing GeoRegistrationPipeline.")

        self.prior_location = prior_location
        self.tile_size_meters = tile_size_meters
        self.hierarchical = hierarchical

        self.outputs_path = Path("sfm/outputs-aachen/")
        self.dataset_path = Path('datasets/aachen/')
        self.images_path = self.dataset_path / 'images/images_upright/'

        self.calibdb_path = self.outputs_path / "calibration.db"
        self.locdb_path = self.outputs_path / "localization.db"

        self.fig_path =  self.outputs_path / 'figs'
        self.fig_path.mkdir(exist_ok=True, parents=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialiaze calibrator and localizer model
        self.calibrator = GeoCalib(weights='pinhole').to(self.device)
        self.localizer = OrienterNetv2(prior_exp_or_path=None, num_rotations=128, device=self.device)

        # Load the existing Aachen SfM model
        self.rec = pycolmap.Reconstruction('sfm/outputs-aachen/colmap_from_nvm_output/')
        
        # Setup image list
        # np.random.shuffle(img_list)
        self.images = list(self.rec.images.values())[:num_images]

        logger.info("GeoRegistrationPipeline initialized.")

        
    def visualize_dataset(self, num_rows):
        img_list = np.array([img for img in os.listdir(self.images_path / 'db/')])
        for i in range(num_rows):
            plot_images([np.asarray(Image.open(os.path.join(self.images_path, 'db', im))) for im in img_list[i*7:i*7+7]])
            p = self.fig_path / 'data' / f"data_{i}.png"
            save_plot(p)
            plt.close()
        return

    def visualize_calib(self):
        logger.info(f"Visualizing calibration results...")
        calibdb = LocalizationDatabase.connect(self.calibdb_path, read_only=True)
        for i,image in enumerate(self.images):
            img_id, geocalib_camera_db, geocalib_gravity_db, _ = calibdb.get_calibration(image.image_id)
            geocalib_camera_db = Camera(geocalib_camera_db).to(self.device)
            geocalib_gravity_db = Gravity(geocalib_gravity_db).to(self.device)
            name = image.name
            img = self.calibrator.load_image(self.images_path / name).to(self.device)
            plot_images([img.permute(1, 2, 0).cpu().numpy()])
            ax = plt.gcf().axes
            plot_perspective_fields([geocalib_camera_db], [geocalib_gravity_db], axes=[ax[0]])
            p = self.fig_path / 'calib' / f"calib_{img_id}_{Path(name).stem}.png"
            save_plot(p)
            plt.close()
        calibdb.close()
        logger.info(f"Saved calib images to {self.fig_path}.")            
        return

    
    def calibrate(self):
        """Estimate gravity direction using GeoCalib"""
        logger.info(f"Calibrating images...")
        if self.calibdb_path.exists():
            # raise FileExistsError("ERROR: database path already exists -- will not modify it.")
            logger.critical("ERROR: database path already exists -- deleting it")
            if os.path.exists(self.calibdb_path):
                os.remove(self.calibdb_path)
        calibdb = LocalizationDatabase.connect(self.calibdb_path)
        calibdb.create_calibration_table()

        for i,image in tqdm(enumerate(self.images)):
            name = image.name
            camera = self.rec.cameras[image.camera_id]

            # read aachen intrinsics
            K = camera.calibration_matrix()
            params = camera.params
            cam_R_world = image.cam_from_world.rotation.matrix()
            cam_t_world = image.cam_from_world.translation.T
            world_R_cam = cam_R_world.T
            world_t_cam = - (world_R_cam @ cam_t_world)
            world_T_cam = Transform3D.from_Rt(world_R_cam, world_t_cam)

            # Estimate roll, pitch, and optionally distortion.
            img = self.calibrator.load_image(self.images_path / name).to(self.device)
            prior_focal = torch.tensor(camera.focal_length).to(self.device)
            calibration_results = self.calibrator.calibrate(img, camera_model='pinhole', priors={'focal': prior_focal})
            # print(f"Results for {name}: {geocalib_results.keys()}")
            # print([val.shape for val in geocalib_results.values()])
            
            estimated_camera = calibration_results['camera']._data.cpu().numpy()
            estimated_gravity = calibration_results['gravity']._data.cpu().numpy()
            gravity_uncertainty = calibration_results['gravity_uncertainty'].squeeze().cpu().numpy()

            # Write to calibration db
            calibdb.add_calibration(image.image_id, estimated_camera, estimated_gravity, gravity_uncertainty)

            # # (visualize geocalib)
            # plot_images([img.permute(1, 2, 0).cpu().numpy()])
            # ax = plt.gcf().axes
            # plot_perspective_fields([geocalib_results["camera"][0]], [geocalib_results["gravity"][0]], axes=[ax[0]])
            # # plot_confidences([results[f"{k}_confidence"][0] for k in ["up", "latitude"]], axes=ax[1:])
            # plt.show()
        
        calibdb.commit()
        calibdb.close()
        logger.info(f"Completed writing {i+1} images into calibdb")
        return


    def localize(self):
        """Estimate 3dof pose using OrienterNetv2"""

        logger.info(f"Localizing images...")
        # debugging: delete existing db
        if self.locdb_path.exists():
            # raise FileExistsError("ERROR: locdb database path already exists -- will not modify it.")
            logger.critical("ERROR: locdb database path already exists -- deleting it")
            if os.path.exists(self.locdb_path):
                os.remove(self.locdb_path)

        calibdb = LocalizationDatabase.connect(self.calibdb_path, read_only=True)
        locdb = LocalizationDatabase.connect(self.locdb_path)
        locdb.create_localization_table()
        localizer = self.localizer
        
        gravities = []
        weights = []
        
        # first, compute the average gravity:
        for i, image in enumerate(self.images):
            name = image.name
            camera = self.rec.cameras[image.camera_id]
            image_path = self.images_path / name

            # read aachen extrinsics
            cam_R_sfm = image.cam_from_world.rotation.matrix()
            cam_t_sfm = image.cam_from_world.translation.T
            sfm_R_cam = cam_R_sfm.T
            sfm_t_cam = - (sfm_R_cam @ cam_t_sfm)
            sfm_T_cam = Transform3D.from_Rt(sfm_R_cam, sfm_t_cam)

            img_id, camera_db, cam_T_gravity, gravity_uncertainty = calibdb.get_calibration(image.image_id)
            estimated_camera = Camera(camera_db).to(self.device)
            # estimated_camera2 = Camera2(estimated_camera._data).to(self.device) # todo: cleanup

            # gravities.append(sfm_T_cam @ cam_T_gravity) # but what we need is gravity_T_cam
            gravities.append(torch.tensor(cam_T_gravity).float().unsqueeze(0))
            weights.append(1/gravity_uncertainty)

        # gravity_weighted_avg = (torch.cat(gravities) * weights).mean(0)
        gravity_weighted_avg = torch.cat(gravities) * torch.cat([torch.from_numpy(weight) for weight in weights]).unsqueeze(-1)
        gravity_weighted_avg = gravity_weighted_avg.mean(0)
        estimated_gravity = Gravity(gravity_weighted_avg).to(self.device)
        roll_pitch = estimated_gravity.rp
        logger.info(f"Estimated roll, pitch = {roll_pitch}")
            
        for i,image in enumerate(self.images):

            name = image.name
            camera = self.rec.cameras[image.camera_id]
            image_path = self.images_path / name

            img_id, sfm_T_cam, _ , _ = calibdb.get_calibration(image.image_id)
            estimated_camera = Camera(sfm_T_cam).to(self.device)
            estimated_camera2 = Camera2(estimated_camera._data).to(self.device) # todo: cleanup
            # estimated_gravity = Gravity(gravity_db).to(self.device)
            # roll_pitch = estimated_gravity.rp
            # logger.info(f"Estimated roll, pitch = {roll_pitch}")

            # Localization: OrienterNetv2 - hierarchical - for large area score volume
            # self.localizer.run(
            #     image_path=self.images_path/name,
            #     prior_address=self.prior_location,
            #     tile_size_meters=self.tile_size_meters,
            #     out_dir=self.fig_path,
            #     hierarchical=True,
            #     topk=3
            # )

            # if hierarchical:
            # assert isinstance(topk, int) and 0 < topk <= 10

            # Read image and create bbox from address
            image_processed, _, _, proj, bbox = preprocess_inputs(
                image_path,
                prior_address=self.prior_location,
                fov=estimated_camera.vfov.cpu(),
                tile_size_meters=self.tile_size_meters,  # try 64, 256, etc.
                calibrate=False
            )

            width = bbox.size[0]
            data = localizer.prepare_inputs(
                image_processed, estimated_camera2.to('cpu'), proj, bbox, hierarchical=self.hierarchical, roll_pitch=torch.rad2deg(roll_pitch).cpu().numpy()
            )
            logger.info("Finished preparing model inputs. Starting inference...")
            # data["camera"] = data['camera'].unsqueeze(0)
            data['camera'] = Camera2(data['camera']._data[:6])
            # data, pred = localizer.localize_hierarchical(data, 2)
            pred = localizer.localize(data)

            if self.hierarchical:
                if localizer.prior_config is not None:
                    key = localizer.prior_config.data.z_max[0]
                    ppm = localizer.prior_model.conf.pixel_per_meter[0]
                else:
                    key = localizer.config.data.z_max[-1] # this is actually 32.0
                    # NOTE: in this case, the fine branch's results are not included/combined
                    ppm = localizer.model.conf.pixel_per_meter[-1]
            else:
                key = "chain"
                ppm = localizer.model.conf.pixel_per_meter[-1]

            # we know the ppm, we know the tile size meters, we know what to expect from the size
                

            logprobs = pred[key]['log_probs']

            if not self.hierarchical:
                # if exhaustive, chain is of fine res. Requires downsampling
                h = w = self.tile_size_meters*2*ppm
                logprobs = torch.nn.functional.interpolate(
                            logprobs.moveaxis(-1, -3), size=(int(h), int(w)), mode="bilinear"
                        ).moveaxis(-3, -1)

            logger.info("Localization inference complete. Preparing visualizations...")
            _, outputs_plot, _, coordinates = localizer.plot_results(data, pred, proj, out_dir=self.fig_path, image_path=image_path)

            # save the scores tensor to disk
            logprobs_path = self.outputs_path / Path('logprobs') / (Path(name).stem + '.pkl')
            plt_path = self.outputs_path / Path('plts') / (Path(name).stem + '.pkl')
            torch.save(logprobs, logprobs_path)
            with open(plt_path, 'wb') as f:
                pickle.dump(outputs_plot, f)

            # # plot_images(inputs_plot)
            # plt.figure(inputs_plot.number)
            # p = self.fig_path / f"localization_INPUT_{img_id}_{Path(name).stem}.png"
            # save_plot(p)
            # plt.close()
            # plot_images(outputs_plot)
            plt.figure(outputs_plot.number)
            p = self.fig_path / "localization" / f"localization_{img_id}_{Path(name).stem}_{coordinates}.png"
            save_plot(p)
            plt.close()
            logger.info(f"Visualizations saved to {self.fig_path}")

            locdb.add_localization(image.image_id, str(logprobs_path))
            logger.info(f"Localization complete.")
            
        calibdb.close()
        locdb.commit()
        locdb.close()
        logger.info(f"Completed writing {i+1} images into locdb")
        return


    def estimate_pose_ransac(self):

        """Given calibrated and localized images in the database,
        return a Transformation that aligns the sfm model with 2D maps,
        and the georegistration (lat lon coordinates) of the images"""
        
        # Load all localization results to memory.
        # Connect to locdb.
        locdb = LocalizationDatabase.connect(self.locdb_path, read_only=True)
        # this should result in a num_cameras x map height x map width x num_rotations
        
        sfm_T_cams = []
        
        debug_cam_t = None

        for i, image in enumerate(self.images):
        
            # Read localization score volume from disk.
            _, logprobs_path = locdb.get_localization(image.image_id)
            logprobs = torch.load(logprobs_path, weights_only=False)
            print(f"Loaded logprobs. Shape: {logprobs.shape}")
            # Stack on top of each other along axis 0
        
        
            # sample transforms ransac - with kabsch

                # this should return 10k poses. Visualize these.

            # score poses. This should be simple, using non-normalized scores.
            
            
            # once we have the right pose, we will need to plot all the cameras at this pose.

            
                # Transform all cameras.
                
                # the cameras have a relative pose, but we do not know the origin. let us assume the origin is the middle of the aachen . This would be roughly about
            #  read aachen extrinsics
            cam_R_sfm = image.cam_from_world.rotation.matrix()
            cam_t_sfm = image.cam_from_world.translation.T
            sfm_R_cam = cam_R_sfm.T # todo: clean
            sfm_t_cam = - (sfm_R_cam @ cam_t_sfm)
            if debug_cam_t is None:
                debug_cam_t = sfm_t_cam
            sfm_t_cam -= debug_cam_t
            sfm_T_cam3d = Transform3D.from_Rt(sfm_R_cam, sfm_t_cam)
            sfm_T_cam = Transform2D.camera_2d_from_3d(sfm_T_cam3d)

            sfm_T_cams.append(sfm_T_cam)


        world_T_guessaachen = Transform2D.from_degrees(torch.tensor([0.]).unsqueeze(0).float(), torch.tensor([256, 256]).unsqueeze(0).float()) # TODO: predict this
        aachen_T_cams = torch.stack(sfm_T_cams).float()
        # world_T_guessaachen = world_T_guessaachen.tile((aachen_T_cams.shape[0],1))
        world_T_cams = world_T_guessaachen.float() @ aachen_T_cams.float()

        # Plot poses. plot_pose
        
        plt_path = logprobs_path.replace('logprobs', 'plts')
        with open(plt_path, 'rb') as f:    
            fig = pickle.load(f)

        plt.figure(fig.number)
        # axes = fig.get_axes()
        axes = fig.axes

        ax = axes[1]
        for i in range(world_T_cams.shape[0]):
            plot_pose(
                ax,
                world_T_cams.t[i],
                world_T_cams.angle[i],
                c="k",
                refactored=True,
                dot=False,
                # s=side / 256,
            )

        p = self.fig_path / "georegistered_aachen" / f"{len(self.images)}.png"
        save_plot(p)

        
        # Get a rough idea of what the GT pose is.

        # Get the pose prediction to work.

        locdb.close()
        logger.info(f"")

        # return best pose
        
        return
        


if __name__ == "__main__":
    pipeline = GeoRegistrationPipeline(
        prior_location="Aachen Cathedral",
        tile_size_meters=256,
        num_images=2
        )

    # pipeline.calibrate()
    # pipeline.visualize_calib()

    # pipeline.localize()

    
    pipeline.estimate_pose_ransac()

    # pipeline.visualize_dataset(num_rows=10)
    
    # pipeline.run()