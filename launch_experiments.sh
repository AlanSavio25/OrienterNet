#!/bin/bash
#SBATCH --job-name=12_2_snap_multiscale_cont
#SBATCH --output=sbatch_outputs/12_2_snap_multiscale_cont.out
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=18
#SBATCH --mem-per-cpu=18G
#SBATCH --account=ls_polle
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gres=gpumem:24G
#SBATCH --signal=INT@600

# nvidia_ge

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

# 3. SNAP coarse - 4x - 2mpp # EXPERIMENT_NAME="snap_coarse_128m_2mpp"
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


# 10_2 Multi-scale Training
# EXPERIMENT_NAME="10_2_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,128,160] \
#         data.max_init_error=[48,96,96] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[4,2,1] \
#         data.tiles_filename=tiles.pkl \
#         model.map_encoder.scale_factor=[1,1,0.5] \
#         model.map_encoder.backbone.output_scales=[0,1,1] \
#         model.pixel_per_meter=[2,1,0.5] \
#         model.pixel_per_meter=[2.0,1.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,1,2] \
#         model.bev_mapper.x_max=[32.0,64.0,128.0] \
#         model.bev_mapper.z_max=[32.0,64.0,128.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

        # data.scenes=[amsterdam] \

# 10_3 => fixed mask pad
# EXPERIMENT_NAME="10_3_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,128,160] \
#         data.max_init_error=[48,96,96] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[1,2,4] \
#         model.pixel_per_meter=[2,1,0.5] \
#         data.tiles_filename=tiles.pkl \
#         model.map_encoder.scale_factor=[1,1,0.5] \
#         model.map_encoder.backbone.output_scales=[0,1,1] \
#         model.pixel_per_meter=[2.0,1.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,1,2] \
#         model.bev_mapper.x_max=[32.0,64.0,128.0] \
#         model.bev_mapper.z_max=[32.0,64.0,128.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000


# 10_4 => Changed BEV NET from Res Block + Adaptation to simple MLP.
# EXPERIMENT_NAME="10_4_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,128,160] \
#         data.max_init_error=[48,96,96] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[1,2,4] \
#         model.pixel_per_meter=[2,1,0.5] \
#         data.tiles_filename=tiles.pkl \
#         model.map_encoder.scale_factor=[1,1,0.5] \
#         model.map_encoder.backbone.output_scales=[0,1,1] \
#         model.pixel_per_meter=[2.0,1.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,1,2] \
#         model.bev_mapper.x_max=[32.0,64.0,128.0] \
#         model.bev_mapper.z_max=[32.0,64.0,128.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.bev_net.num_blocks=0 \
#         model.bev_mapper.bev_net.mlp.layers=[128,128,8] \
#         model.bev_mapper.bev_net.mlp.input_dim=128 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

## Single Random Scale: SHARING OUTPUT LEVEL [Image Features, Neural Map]
# 10_5 => Fix BEV Net (MLP) and MAP (MaxPool) [has artifacts, BEV net is slightly weaker than 10_6]
# EXPERIMENT_NAME="10_5_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,128,160] \
#         data.max_init_error=[48,96,96] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[1,2,4] \
#         model.pixel_per_meter=[2,1,0.5] \
#         data.tiles_filename=tiles.pkl \
#         model.map_encoder.backbone.output_scales=[0] \
#         model.map_encoder.max_pool_ksize=[1,2,4] \
#         model.pixel_per_meter=[2.0,1.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,1,2] \
#         model.bev_mapper.x_max=[32.0,64.0,128.0] \
#         model.bev_mapper.z_max=[32.0,64.0,128.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.bev_net.num_blocks=0 \
#         model.bev_mapper.bev_net.mlp.layers=[128,128,8] \
#         model.bev_mapper.bev_net.mlp.input_dim=128 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

        # model.map_encoder.scale_factor=[1,1,1] \

## Single Random Scale: SHARING OUTPUT LEVEL [Image Features, Neural Map]
# 10_6 => Fix max_init_error (artifacts), update BEV Net (MLP)
# EXPERIMENT_NAME="10_6_snap_multiscale_test"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,128,160] \
#         data.max_init_error=[48,96,96] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[1,2,4] \
#         model.map_encoder.backbone.output_scales=[0] \
#         model.map_encoder.max_pool_ksize=[1,2,4] \
#         model.pixel_per_meter=[2.0,1.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,1,2] \
#         model.bev_mapper.x_max=[32.0,64.0,128.0] \
#         model.bev_mapper.z_max=[32.0,64.0,128.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.bev_net.num_blocks=0 \
#         model.bev_mapper.bev_net.mlp.layers=[256,128] \
#         model.bev_mapper.bev_net.mlp.input_dim=128 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# 10_6_debug => Fix max_init_error (artifacts), update BEV Net (MLP)
# EXPERIMENT_NAME="10_6_snap_multiscale_test_DEBUG64m"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[128] \
#         data.max_init_error=[96] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[2] \
#         model.map_encoder.backbone.output_scales=[0] \
#         model.map_encoder.max_pool_ksize=[2] \
#         model.pixel_per_meter=[1.0] \
#         model.bev_mapper.grid_cell_size=[1] \
#         model.bev_mapper.x_max=[64.0] \
#         model.bev_mapper.z_max=[64.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.bev_net.num_blocks=0 \
#         model.bev_mapper.bev_net.mlp.layers=[256,128] \
#         model.bev_mapper.bev_net.mlp.input_dim=128 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000


# # 10_7 => Split Features (final layer 2x) and Map (pyramid)
# EXPERIMENT_NAME="10_7_snap_multiscale_test"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,128,160] \
#         data.max_init_error=[48,96,96] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[1,2,4] \
#         model.map_encoder.backbone.output_scales=[0,1,1] \
#         model.map_encoder.max_pool_ksize=[1,1,2] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=256 \
#         model.bev_mapper.feature_map_split_idx=[0,1,1] \
#         model.pixel_per_meter=[2.0,1.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,1,2] \
#         model.bev_mapper.x_max=[32.0,64.0,128.0] \
#         model.bev_mapper.z_max=[32.0,64.0,128.0] \
#         model.bev_mapper.bev_net.num_blocks=0 \
#         model.bev_mapper.bev_net.mlp.layers=[256,128] \
#         model.bev_mapper.bev_net.mlp.input_dim=128 \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# 10_8 => MULTISCALE model. Split image features, Map (pyramid)
# commit 48500c19448aa895f084f46f0e3e6539f81d993e, multiscale_training_multi
# Changed to *local* branch, multiscale_training_multi_exp10_8
# EXPERIMENT_NAME="10_8_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,160] \
#         data.max_init_error=[48,48] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[1,4] \
#         data.add_map_mask=True \
#         model.map_encoder.backbone.output_scales=[0,1] \
#         model.multiscale=True \
#         model.map_encoder.backbone.max_pool_ksize=[1,2] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=256 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# 10_9 => MULTISCALE model. Untie fusion MLP and scale classifier. 
# commit: 
# EXPERIMENT_NAME="10_9_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,160] \
#         data.max_init_error=[48,48] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[1,4] \
#         data.add_map_mask=True \
#         model.map_encoder.backbone.output_scales=[0,1] \
#         model.multiscale=True \
#         model.map_encoder.backbone.max_pool_ksize=[1,2] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=256 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# 10_10 Single Scale - Finer BEV AND Finer Map Raster coarse - 2mpp # Replacement of 9_16
# Unfortunately, this is running from one of the two branches: multiscale_training_multi_exp10_{8,9}
# EXPERIMENT_NAME="10_10_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.return_multiscale=True \
#         data.crop_size_meters=[256] \
#         data.mask_pad=[4] \
#         data.max_init_error=[192] \
#         data.pixel_per_meter=1 \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.multiscale=True \
#         model.map_encoder.backbone.output_scales=[1] \
#         model.map_encoder.backbone.max_pool_ksize=[1] \
#         model.pixel_per_meter=[0.5] \
#         model.bev_mapper.z_max=[128.0] \
#         model.bev_mapper.x_max=[128.0] \
#         model.bev_mapper.grid_cell_size=[1.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         data.z_max=[128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# 10_11 standard coarse - 2mpp # For comparing directly with previous (which should outperform this) # replacement of 9_9
# This is sort of a confirmation that things are identical after the new refactoring
# EXPERIMENT_NAME="10_11_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.return_multiscale=True \
#         data.crop_size_meters=[256] \
#         data.mask_pad=[4] \
#         data.max_init_error=[192] \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.multiscale=True \
#         model.map_encoder.backbone.output_scales=[0] \
#         model.map_encoder.backbone.max_pool_ksize=[1] \
#         model.pixel_per_meter=[0.5] \
#         model.bev_mapper.z_max=[128.0] \
#         model.bev_mapper.x_max=[128.0] \
#         model.bev_mapper.grid_cell_size=[2.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         data.z_max=[128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# 10_12 => MULTISCALE model. Same as 10_9, except now we share features.
# This experiment tells us whether tying the features together harms performance or not.
# EXPERIMENT_NAME="10_12_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,160] \
#         data.max_init_error=[48,48] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[1,4] \
#         data.add_map_mask=True \
#         model.map_encoder.backbone.output_scales=[0,1] \
#         model.multiscale=True \
#         model.map_encoder.backbone.max_pool_ksize=[1,2] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# MAP DOWNSAMPLE again - this time repeat of before (except now we have max pool before the final computation)
# We are also experimenting with the coarse64m map downsample
# 10_13 SNAP coarse - 2x - 1mpp
# EXPERIMENT_NAME="10_13_snap_coarse_64m_1mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.return_multiscale=True \
#         data.crop_size_meters=[128] \
#         data.max_init_error=[96] \
#         data.mask_pad=[4] \
#         data.pixel_per_meter=2 \
#         model.pixel_per_meter=[1] \
#         model.multiscale=True \
#         data.tiles_filename=tiles.pkl \
#         model.map_encoder.backbone.output_scales=[1] \
#         model.map_encoder.backbone.max_pool_ksize=[1] \
#         model.bev_mapper.z_max=[64.0] \
#         model.bev_mapper.x_max=[64.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.grid_cell_size=[1.0] \
#         data.z_max=[64.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# 10_14 SNAP coarse - 4x - 2mpp
# EXPERIMENT_NAME="10_14_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.return_multiscale=True \
#         data.crop_size_meters=[256] \
#         data.max_init_error=[192] \
#         data.mask_pad=[6] \
#         data.pixel_per_meter=1 \
#         model.pixel_per_meter=[0.5] \
#         model.multiscale=True \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.map_encoder.backbone.output_scales=[1] \
#         model.map_encoder.backbone.max_pool_ksize=[1] \
#         model.bev_mapper.z_max=[128.0] \
#         model.bev_mapper.x_max=[128.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.grid_cell_size=[2.0] \
#         data.z_max=[128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000


# 10_15 SNAP coarse - 4x - 2mpp - Finer BEV with conv instead of bilinear downsampling
# EXPERIMENT_NAME="10_15_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.return_multiscale=True \
#         data.crop_size_meters=[256] \
#         data.max_init_error=[192] \
#         data.mask_pad=[4] \
#         data.pixel_per_meter=0.5 \
#         model.pixel_per_meter=[0.5] \
#         model.multiscale=True \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.map_encoder.backbone.output_scales=[0] \
#         model.map_encoder.backbone.max_pool_ksize=[1] \
#         model.bev_mapper.z_max=[128.0] \
#         model.bev_mapper.x_max=[128.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.grid_cell_size=[1.0] \
#         data.z_max=[128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# 10_16 SNAP coarse - 4x - 2mpp - Finer BEV (fixed) AND Finer MAP
# EXPERIMENT_NAME="10_16_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.return_multiscale=True \
#         data.crop_size_meters=[256] \
#         data.max_init_error=[192] \
#         data.mask_pad=[6] \
#         data.pixel_per_meter=1 \
#         model.pixel_per_meter=[0.5] \
#         model.multiscale=True \
#         data.tiles_filename=tiles_1mpp.pkl \
#         model.map_encoder.backbone.output_scales=[1] \
#         model.map_encoder.backbone.max_pool_ksize=[1] \
#         model.bev_mapper.z_max=[128.0] \
#         model.bev_mapper.x_max=[128.0] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.grid_cell_size=[1.0] \
#         data.z_max=[128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000

# 11_0 => MULTISCALE model. Same as 10_9 but with larger coarse map
# commit: 
# EXPERIMENT_NAME="11_0_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_1mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,224] \
#         data.max_init_error=[48,192] \
#         data.pixel_per_meter=1 \
#         data.mask_pad=[2,4] \
#         data.add_map_mask=True \
#         model.map_encoder.backbone.output_scales=[0,0] \
#         model.multiscale=True \
#         model.map_encoder.backbone.scale_factor=[2,0.5] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=256 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4

# # 11_1 => Re-running 9_7. We might have an issue somewhere
# #  SNAP fine - 1x - 0.5mpp
# EXPERIMENT_NAME="11_1_snap_fine"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64] \
#         data.max_init_error=[48] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[1] \
#         data.add_map_mask=True \
#         model.map_encoder.backbone.output_scales=[0] \
#         model.map_encoder.unary_prior=False \
#         model.multiscale=True \
#         model.map_encoder.backbone.scale_factor=[1] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.pixel_per_meter=[2] \
#         model.bev_mapper.grid_cell_size=[0.5] \
#         model.bev_mapper.x_max=[32.0] \
#         model.bev_mapper.z_max=[32.0] \
#         training.lr=0.0001 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4

# # 11_2 => Re-running 9_9. We might have an issue somewhere
# #  SNAP fine - 1x - 0.5mpp
# EXPERIMENT_NAME="11_2_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_2mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[256] \
#         data.max_init_error=[192] \
#         data.pixel_per_meter=0.5 \
#         data.mask_pad=[4] \
#         data.add_map_mask=True \
#         model.map_encoder.backbone.output_scales=[0] \
#         model.map_encoder.unary_prior=False \
#         model.multiscale=True \
#         model.map_encoder.backbone.scale_factor=[1] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.pixel_per_meter=[0.5] \
#         model.bev_mapper.grid_cell_size=[2] \
#         model.bev_mapper.x_max=[128.0] \
#         model.bev_mapper.z_max=[128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4



# # 11_3 => 1_mpp map but fine model 
# #  SNAP fine - 1x - 0.5mpp. Map at 1mpp
# EXPERIMENT_NAME="11_3_snap_fine"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_1mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64] \
#         data.max_init_error=[48] \
#         data.pixel_per_meter=1 \
#         data.mask_pad=[2] \
#         data.add_map_mask=True \
#         model.map_encoder.backbone.output_scales=[0] \
#         model.map_encoder.unary_prior=False \
#         model.multiscale=True \
#         model.map_encoder.backbone.scale_factor=[2] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.pixel_per_meter=[2] \
#         model.bev_mapper.grid_cell_size=[0.5] \
#         model.bev_mapper.x_max=[32.0] \
#         model.bev_mapper.z_max=[32.0] \
#         training.lr=0.0001 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4


# # 11_4 => MULTISCALE model.
# EXPERIMENT_NAME="11_4_snap_multiscale_SPLIT"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_1mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,224] \
#         data.max_init_error=[48,192] \
#         data.pixel_per_meter=1 \
#         data.mask_pad=[2,4] \
#         data.add_map_mask=True \
#         model.multiscale=True \
#         model.map_encoder.backbone.output_scales=[0,0] \
#         model.map_encoder.backbone.scale_factor=[2,0.5] \
#         model.map_encoder.backbone.num_branches=2 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.bev_mapper.image_encoder.backbone.num_branches=2 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \

# # 11_5 => MULTISCALE model. 
# EXPERIMENT_NAME="11_5_snap_multiscale_SPLITMAP"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_1mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,224] \
#         data.max_init_error=[48,192] \
#         data.pixel_per_meter=1 \
#         data.mask_pad=[2,4] \
#         data.add_map_mask=True \
#         model.multiscale=True \
#         model.map_encoder.backbone.output_scales=[0,0] \
#         model.map_encoder.backbone.scale_factor=[2,0.5] \
#         model.map_encoder.backbone.num_branches=2 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=256 \
#         model.bev_mapper.image_encoder.backbone.num_branches=1 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \

# # 11_6 => MULTISCALE model. 
# EXPERIMENT_NAME="11_6_snap_multiscale_SPLITIMAGE"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_1mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,224] \
#         data.max_init_error=[48,192] \
#         data.pixel_per_meter=1 \
#         data.mask_pad=[2,4] \
#         data.add_map_mask=True \
#         model.multiscale=True \
#         model.map_encoder.backbone.output_scales=[0,0] \
#         model.map_encoder.backbone.scale_factor=[2,0.5] \
#         model.map_encoder.backbone.num_branches=1 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.bev_mapper.image_encoder.backbone.num_branches=2 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \

# # 11_9  - added map augmentations
# #  SNAP fine - 1x - 0.5mpp
# EXPERIMENT_NAME="11_9_snap_fine"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64] \
#         data.max_init_error=[48] \
#         data.pixel_per_meter=2 \
#         data.mask_pad=[1] \
#         data.add_map_mask=True \
#         model.map_encoder.backbone.output_scales=[0] \
#         model.map_encoder.unary_prior=False \
#         model.multiscale=True \
#         model.map_encoder.backbone.scale_factor=[1] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.pixel_per_meter=[2] \
#         model.bev_mapper.grid_cell_size=[0.5] \
#         model.bev_mapper.x_max=[32.0] \
#         model.bev_mapper.z_max=[32.0] \
#         training.lr=0.0001 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4

# # 11_10, added map augmentations
# # SNAP coarse 2mpp
# EXPERIMENT_NAME="11_10_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_2mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[256] \
#         data.max_init_error=[192] \
#         data.pixel_per_meter=0.5 \
#         data.mask_pad=[4] \
#         data.add_map_mask=True \
#         model.map_encoder.backbone.output_scales=[0] \
#         model.map_encoder.unary_prior=False \
#         model.multiscale=True \
#         model.map_encoder.backbone.scale_factor=[1] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.pixel_per_meter=[0.5] \
#         model.bev_mapper.grid_cell_size=[2] \
#         model.bev_mapper.x_max=[128.0] \
#         model.bev_mapper.z_max=[128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4


# # 11_11, added map augmentations
# # SNAP coarse 2mpp
# EXPERIMENT_NAME="11_11_snap_coarse_128m_2mpp_48m"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_2mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[256] \
#         data.max_init_error=[48] \
#         data.pixel_per_meter=0.5 \
#         data.mask_pad=[4] \
#         data.add_map_mask=True \
#         model.map_encoder.backbone.output_scales=[0] \
#         model.map_encoder.unary_prior=False \
#         model.multiscale=True \
#         model.map_encoder.backbone.scale_factor=[1] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.pixel_per_meter=[0.5] \
#         model.bev_mapper.grid_cell_size=[2] \
#         model.bev_mapper.x_max=[128.0] \
#         model.bev_mapper.z_max=[128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4


# # 11_12 => Multiscale model, Split Image and Map encoder, without temperature
# EXPERIMENT_NAME="11_12_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_1mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,160] \
#         data.max_init_error=[48,48] \
#         data.pixel_per_meter=1 \
#         data.mask_pad=[4,4] \
#         data.add_map_mask=True \
#         model.multiscale=True \
#         model.add_temperature=False \
#         model.rescale_coarser_prob=True \
#         model.map_encoder.backbone.output_scales=[0,0] \
#         model.map_encoder.backbone.scale_factor=[2,0.5] \
#         model.map_encoder.backbone.num_branches=2 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.bev_mapper.image_encoder.backbone.num_branches=2 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \

# exit 0 

# 11_{13,14,18,20} => Multiscale model, Split Image and Map encoder, with temperature - same exp
# EXPERIMENT_NAME="11_20_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_1mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,160] \
#         data.max_init_error=[48,48] \
#         data.pixel_per_meter=1 \
#         data.mask_pad=[4,4] \
#         data.add_map_mask=True \
#         model.multiscale=True \
#         model.add_temperature=True \
#         model.rescale_coarser_prob=True \
#         model.map_encoder.backbone.output_scales=[0,0] \
#         model.map_encoder.backbone.scale_factor=[2,0.5] \
#         model.map_encoder.backbone.num_branches=2 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.bev_mapper.image_encoder.backbone.num_branches=2 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \

# # 11_15 => Multiscale model, Split Image and Map encoder, same as 11_12, except larger map, and no rescaling prob
# EXPERIMENT_NAME="11_15_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_1mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,224] \
#         data.max_init_error=[48,192] \
#         data.pixel_per_meter=1 \
#         data.mask_pad=[4,4] \
#         data.add_map_mask=True \
#         model.multiscale=True \
#         model.add_temperature=False \
#         model.rescale_coarser_prob=False \
#         model.map_encoder.backbone.output_scales=[0,0] \
#         model.map_encoder.backbone.scale_factor=[2,0.5] \
#         model.map_encoder.backbone.num_branches=2 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.bev_mapper.image_encoder.backbone.num_branches=2 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \

# # 11_{16,19,21} => Multiscale model, Split Image and Map encoder, same as 11_14, except larger map, and no rescaling prob
# EXPERIMENT_NAME="11_19_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_1mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,224] \
#         data.max_init_error=[48,192] \
#         data.pixel_per_meter=1 \
#         data.mask_pad=[4,4] \
#         data.add_map_mask=True \
#         model.multiscale=True \
#         model.add_temperature=True \
#         model.rescale_coarser_prob=False \
#         model.map_encoder.backbone.output_scales=[0,0] \
#         model.map_encoder.backbone.scale_factor=[2,0.5] \
#         model.map_encoder.backbone.num_branches=2 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.bev_mapper.image_encoder.backbone.num_branches=2 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \


# # # 11_17 => Multiscale model, Single Decoder, Split Map encoder. Larger map, no rescaling prob, no temperature, same as 11_15, except single decoder
# EXPERIMENT_NAME="11_17_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_1mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,224] \
#         data.max_init_error=[48,192] \
#         data.pixel_per_meter=1 \
#         data.mask_pad=[4,4] \
#         data.add_map_mask=True \
#         model.multiscale=True \
#         model.add_temperature=False \
#         model.rescale_coarser_prob=False \
#         model.map_encoder.backbone.output_scales=[0,0] \
#         model.map_encoder.backbone.scale_factor=[2,0.5] \
#         model.map_encoder.backbone.num_branches=2 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=256 \
#         model.bev_mapper.image_encoder.backbone.num_branches=1 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \


# 11_22 => Multiscale model, Single Decoder, Split Map encoder. Larger map, no rescaling prob, WITH temperature, same as 11_15, except single decoder
# EXPERIMENT_NAME="11_22_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_1mpp.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,224] \
#         data.max_init_error=[48,192] \
#         data.pixel_per_meter=1 \
#         data.mask_pad=[4,4] \
#         data.add_map_mask=True \
#         model.multiscale=True \
#         model.add_temperature=True \
#         model.rescale_coarser_prob=False \
#         model.map_encoder.backbone.output_scales=[0,0] \
#         model.map_encoder.backbone.scale_factor=[2,0.5] \
#         model.map_encoder.backbone.num_branches=2 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=256 \
#         model.bev_mapper.image_encoder.backbone.num_branches=1 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \


# # 12_0
# # SNAP coarse 2mpp
# EXPERIMENT_NAME="12_0_snap_coarse_256m"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=tiles_2mpp_extended.pkl \
#         data.return_multiscale=True \
#         data.crop_size_meters=[512] \
#         data.max_init_error=[384] \
#         data.pixel_per_meter=0.5 \
#         data.mask_pad=[8] \
#         data.add_map_mask=True \
#         model.map_encoder.backbone.output_scales=[1] \
#         model.map_encoder.unary_prior=False \
#         model.multiscale=True \
#         model.map_encoder.backbone.scale_factor=[1] \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=128 \
#         model.pixel_per_meter=[0.25] \
#         model.bev_mapper.grid_cell_size=[4] \
#         model.bev_mapper.x_max=[256.0] \
#         model.bev_mapper.z_max=[256.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \
#         # training.trainer.val_check_interval=50

# # 12_1 => Multiscale (32m,128m) 2 separate map encoders.
# EXPERIMENT_NAME="12_1_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=[tiles.pkl,tiles_1mpp.pkl] \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,256] \
#         data.max_init_error=[48,192] \
#         data.pixel_per_meter=[2,1] \
#         data.mask_pad=[1,4] \
#         data.add_map_mask=True \
#         model.multiscale=True \
#         model.add_temperature=True \
#         model.map_encoder.num_encoders=2 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=256 \
#         model.bev_mapper.image_encoder.backbone.num_branches=1 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \
#         # model.map_encoder.backbone.output_scales=[0,0] \
#         # model.map_encoder.backbone.scale_factor=[1,1] \
#         # model.map_encoder.backbone.num_branches=1 \

# 12_2 => Multiscale (32m,256m) 2 separate map encoders.
EXPERIMENT_NAME="12_2_snap_multiscale"
python -m maploc.train experiment.name=$EXPERIMENT_NAME \
        data.tiles_filename=[tiles.pkl,tiles_2mpp_extended.pkl] \
        data.return_multiscale=True \
        data.crop_size_meters=[64,512] \
        data.max_init_error=[48,384] \
        data.pixel_per_meter=[2,0.5] \
        data.mask_pad=[1,8] \
        data.add_map_mask=True \
        model.multiscale=True \
        model.add_temperature=True \
        model.map_encoder.num_encoders=2 \
        model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
        model.bev_mapper.image_encoder.backbone.output_dim=256 \
        model.bev_mapper.image_encoder.backbone.num_branches=1 \
        model.pixel_per_meter=[2.0,0.25] \
        model.bev_mapper.grid_cell_size=[0.5,4] \
        model.bev_mapper.x_max=[32.0,256.0] \
        model.bev_mapper.z_max=[32.0,256.0] \
        training.lr=5e-5 \
        training.trainer.max_steps=320000 \
        data.loading.train.batch_size=4 \
        # model.map_encoder.backbone.output_scales=[0,0] \
        # model.map_encoder.backbone.scale_factor=[1,1] \
        # model.map_encoder.backbone.num_branches=1 \
        

# # 12_3 => Multiscale (32m,128m) 2 separate map encoders, finer image features. Couldn't run this because it runs OOM
# # 12_{4,5} => same, except larger crop size changed to 256 from 224
# EXPERIMENT_NAME="12_5_snap_multiscale"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.tiles_filename=[tiles.pkl,tiles_1mpp.pkl] \
#         data.return_multiscale=True \
#         data.crop_size_meters=[64,256] \
#         data.max_init_error=[48,192] \
#         data.pixel_per_meter=[2,1] \
#         data.mask_pad=[1,4] \
#         data.add_map_mask=True \
#         model.multiscale=True \
#         model.add_temperature=True \
#         model.map_encoder.num_encoders=2 \
#         model.bev_mapper.image_encoder.backbone.remove_stride_from_first_conv=True \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         model.bev_mapper.image_encoder.backbone.output_dim=256 \
#         model.bev_mapper.image_encoder.backbone.num_branches=1 \
#         model.pixel_per_meter=[2.0,0.5] \
#         model.bev_mapper.grid_cell_size=[0.5,2] \
#         model.bev_mapper.x_max=[32.0,128.0] \
#         model.bev_mapper.z_max=[32.0,128.0] \
#         training.lr=5e-5 \
#         training.trainer.max_steps=320000 \
#         data.loading.train.batch_size=4 \
#         # model.bev_mapper.image_encoder.backbone.remove_stride_from_first_conv=True \
#         # model.map_encoder.backbone.output_scales=[0,0] \
#         # model.map_encoder.backbone.scale_factor=[1,1] \
#         # model.map_encoder.backbone.num_branches=1 \

exit 0
