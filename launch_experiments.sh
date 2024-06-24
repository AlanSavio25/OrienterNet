#!/bin/bash
#SBATCH --job-name=10_2_snap_multiscale
#SBATCH --output=sbatch_outputs/10_2_snap_multiscale.out
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=14G
#SBATCH --account=ls_polle
#SBATCH --gpus=1
#SBATCH --gres=gpumem:25G
#SBATCH --signal=INT@600

# nvidia_geforce_rtx_4090

# NOTE: if continuing an experiment, then name the sbatch output differently

# 0. OrienterNet fine
# EXPERIMENT_NAME="orienternet_fine"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=True \
#         data.crop_size_meters=64 \
#         data.pixel_per_meter=2 \
#         data.tiles_filename=tiles.pkl \
#         model.bev_mapper.mode=forward \
#         model.bev_mapper.z_max=32.0 \
#         model.bev_mapper.x_max=32.0

# 1. SNAP fine
# EXPERIMENT_NAME="snap_fine"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=64 \
#         data.pixel_per_meter=2 \
#         data.tiles_filename=tiles.pkl \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=32.0 \
#         model.bev_mapper.x_max=32.0 \

# 2. OrienterNet coarse - 4x - 2mpp
# EXPERIMENT_NAME="orienternet_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=True \
#         data.crop_size_meters=256 \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.bev_mapper.mode=forward \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0

# 3. SNAP coarse - 4x - 2mpp
# EXPERIMENT_NAME="snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=256 \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0


# 4. OrienterNet coarse - 2x - 1mpp
# EXPERIMENT_NAME="orienternet_coarse_64m_1mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=True \
#         data.crop_size_meters=128 \
#         data.pixel_per_meter=1 \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.bev_mapper.mode=forward \
#         model.bev_mapper.z_max=64.0 \
#         model.bev_mapper.x_max=64.0

# 5. SNAP coarse - 2x - 1mpp
# EXPERIMENT_NAME="snap_coarse_64m_1mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=128 \
#         data.pixel_per_meter=1 \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=64.0 \
#         model.bev_mapper.x_max=64.0


# ??? 6. OrienterNet coarse - 8x - 4mpp
# EXPERIMENT_NAME="orienternet_coarse_256m_4mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=True \
#         data.crop_size_meters=512 \
#         data.pixel_per_meter=0.25 \
#         data.tiles_filename=tiles_4mpp.pkl \
#         model.bev_mapper.mode=forward \
#         model.bev_mapper.z_max=256.0 \
#         model.bev_mapper.x_max=256.0


# ??? 7.{0,9} SNAP coarse - 8x - 4mpp - unary prior off
# EXPERIMENT_NAME="7_9_snap_coarse_256m_4mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=512 \
#         data.max_init_error=384 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=0.25 \
#         data.tiles_filename=tiles_4mpp.pkl \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=256.0 \
#         model.bev_mapper.x_max=256.0


# 7.1 SNAP coarse - 8x - 4mpp - unary prior on
# EXPERIMENT_NAME="7_1_snap_coarse_256m_4mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=512 \
#         data.max_init_error=384 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=0.25 \
#         data.tiles_filename=tiles_4mpp.pkl \
#         model.map_encoder.unary_prior=True \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=256.0 \
#         model.bev_mapper.x_max=256.0

# 7.2 SNAP coarse - 8x - 4mpp - unary prior on, apply_map_prior on
# EXPERIMENT_NAME="7_2_snap_coarse_256m_4mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=512 \
#         data.max_init_error=384 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=0.25 \
#         data.tiles_filename=tiles_4mpp.pkl \
#         model.map_encoder.unary_prior=True \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=256.0 \
#         model.bev_mapper.x_max=256.0

# 7.3 SNAP fine - 1x - 0.5mpp - unary prior off - MAP MASK ON
# EXPERIMENT_NAME="7_3_snap_fine"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=64 \
#         data.max_init_error=48 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=2 \
#         data.tiles_filename=tiles.pkl \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=32.0 \
#         model.bev_mapper.x_max=32.0


# 7.{4,8} SNAP coarse - 2x - 1mpp
# EXPERIMENT_NAME="7_8_snap_coarse_64m_1mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=128 \
#         data.max_init_error=96 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=1 \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=64.0 \
#         model.bev_mapper.x_max=64.0

# 7_{5,6,7} SNAP coarse - 4x - 2mpp
# EXPERIMENT_NAME="7_7_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=256 \
#         data.max_init_error=192 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0

# 7. SNAP fine - 1x - 0.5mpp - 2x LARGER MAP - crop size 128
# EXPERIMENT_NAME="snap_fineLARGEMAP"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=96 \
#         data.pixel_per_meter=2 \
#         data.tiles_filename=tiles.pkl \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=32.0 \
#         model.bev_mapper.x_max=32.0 \



# === Learning Rate Experiments === #

# 8_0 SNAP coarse - 4x - 2mpp
# EXPERIMENT_NAME="8_0_snap_coarse_128m_2mpp_lr5"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=256 \
#         data.max_init_error=192 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0 \
#         training.lr=1e-5


# 8_0 SNAP coarse - 4x - 2mpp
# EXPERIMENT_NAME="8_1_snap_coarse_128m_2mpp_lr6"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=256 \
#         data.max_init_error=192 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0 \
#         training.lr=1e-6

### == Normalize Features ON === #

# EXPERIMENT_NAME="8_2_snap_coarse_128m_2mpp_NF"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=256 \
#         data.max_init_error=192 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0 \
#         model.normalize_features=True

### === inf mask pad fixed === ###
# Running 128 1e-4, 128 1e-5, 64 1e-4, 256 1e-4

# 8_3, 9_10 SNAP coarse - 4x - 2mpp , lr 1e-4 (default) 
# EXPERIMENT_NAME="9_10_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=256 \
#         data.max_init_error=192 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0

# # 8_4 SNAP coarse - 4x - 2mpp
# EXPERIMENT_NAME="8_4_snap_coarse_128m_2mpp_lr1e-5"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=256 \
#         data.max_init_error=192 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0 \
#         training.lr=1e-5


# EXPERIMENT_NAME="8_5_snap_coarse_64m_1mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=128 \
#         data.max_init_error=96 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=1 \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=64.0 \
#         model.bev_mapper.x_max=64.0


# EXPERIMENT_NAME="8_6_snap_coarse_256m_4mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=False \
#         data.crop_size_meters=512 \
#         data.max_init_error=384 \
#         data.add_map_mask=True \
#         data.pixel_per_meter=0.25 \
#         data.tiles_filename=tiles_4mpp.pkl \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.z_max=256.0 \
#         model.bev_mapper.x_max=256.0



### ---- Last Experiments with lower LR and ResNet-18 --- ###
# All with map mask on, unary prior off

# 9 {0,5,7} SNAP fine - 1x - 0.5mpp # 4090 # disk
# EXPERIMENT_NAME="9_7_snap_fine"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.crop_size_meters=64 \
#         data.max_init_error=48 \
#         data.pixel_per_meter=2 \
#         data.tiles_filename=tiles.pkl \
#         data.add_map_mask=True \
#         model.map_encoder.unary_prior=False \
#         model.bev_mapper.z_max=32.0 \
#         model.bev_mapper.x_max=32.0 \
#         model.bev_mapper.mode=inverse \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=0.0001

# model.bev_mapper.image_encoder.backbone.encoder=resnet18 \

# # 9_{1,4,8,11} SNAP coarse - 2x - 64m # 4090
# EXPERIMENT_NAME="9_11_snap_coarse_64m_1mpp_restart"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.crop_size_meters=128 \
#         data.max_init_error=96 \
#         data.pixel_per_meter=1 \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.bev_mapper.z_max=64.0 \
#         model.bev_mapper.x_max=64.0 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# # 9_{2,3,9,12} SNAP coarse - 4x - 2mpp # 4090
# EXPERIMENT_NAME="9_12_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.crop_size_meters=256 \
#         data.max_init_error=192 \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# # # 9_{13} SNAP coarse - 2x - 64m. Fine BEV -> downsample
# EXPERIMENT_NAME="9_13_snap_coarse_64m_1mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.crop_size_meters=128 \
#         data.max_init_error=96 \
#         data.pixel_per_meter=1 \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.bev_mapper.z_max=64.0 \
#         model.bev_mapper.x_max=64.0 \
#         model.bev_mapper.grid_cell_size=0.5 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# # 9_14 SNAP coarse - 4x - 2mpp
# EXPERIMENT_NAME="9_14_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.crop_size_meters=256 \
#         data.max_init_error=192 \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0 \
#         model.bev_mapper.grid_cell_size=1 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# With align corners=False

# 9_{15} SNAP coarse - 2x - 64m. Fine BEV -> downsample
# EXPERIMENT_NAME="9_15_snap_coarse_64m_1mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.crop_size_meters=128 \
#         data.max_init_error=96 \
#         data.pixel_per_meter=1 \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.bev_mapper.z_max=64.0 \
#         model.bev_mapper.x_max=64.0 \
#         model.bev_mapper.grid_cell_size=0.5 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# # 9_16 SNAP coarse - 4x - 2mpp
# EXPERIMENT_NAME="9_16_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.crop_size_meters=256 \
#         data.max_init_error=192 \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0 \
#         model.bev_mapper.grid_cell_size=1 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000


# MAP DOWNSAMPLE

# 9_17 SNAP coarse - 2x - 1mpp NOT READY
# EXPERIMENT_NAME="9_17_snap_coarse_64m_1mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.crop_size_meters=128 \
#         data.max_init_error=96 \
#         data.pixel_per_meter=1 \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.bev_mapper.z_max=64.0 \
#         model.bev_mapper.x_max=64.0 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# 9_{18} SNAP coarse - 4x - 2mpp
# EXPERIMENT_NAME="9_18_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.crop_size_meters=256 \
#         data.max_init_error=192 \
#         data.mask_pad=6 \
#         data.pixel_per_meter=1 \
#         model.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.map_encoder.backbone.output_scales=[1] \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000


# 10_0 Multi-scale Training
EXPERIMENT_NAME="10_2_snap_multiscale"
python -m maploc.train experiment.name=$EXPERIMENT_NAME \
        data.return_multiscale=True \
        data.crop_size_meters=[64,128,160] \
        data.max_init_error=[48,96,96] \
        data.pixel_per_meter=2 \
        data.mask_pad=[4,2,1] \
        model.pixel_per_meter=[2,1,0.5] \
        data.tiles_filename=tiles.pkl \
        model.map_encoder.scale_factor=[1,1,0.5] \
        model.map_encoder.backbone.output_scales=[0,1,1] \
        model.pixel_per_meter=[2.0,1.0,0.5] \
        model.bev_mapper.grid_cell_size=[0.5,1,2] \
        model.bev_mapper.x_max=[32.0,64.0,128.0] \
        model.bev_mapper.z_max=[32.0,64.0,128.0] \
        model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
        training.lr=5e-5 \
        training.trainer.max_steps=320000

        # data.scenes=[amsterdam] \

exit 0
