#!/bin/bash
#SBATCH --job-name=9_1_snap_coarse_64m_1mpp
#SBATCH --output=sbatch_outputs/9_1_snap_coarse_64m_1mpp.out
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=12G
#SBATCH --account=ls_polle
#SBATCH --gpus=rtx_3090:1
#SBATCH --gres=gpumem:20G
#SBATCH --signal=INT@600

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

# 8_3 SNAP coarse - 4x - 2mpp , lr 1e-4 (default) 
# EXPERIMENT_NAME="8_3_snap_coarse_128m_2mpp"
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

# 9.0 SNAP fine - 1x - 0.5mpp
# EXPERIMENT_NAME="9_0_snap_fine"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.crop_size_meters=64 \
#         data.max_init_error=48 \
#         data.pixel_per_meter=2 \
#         data.tiles_filename=tiles.pkl \
#         model.bev_mapper.z_max=32.0 \
#         model.bev_mapper.x_max=32.0 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=1e-4

# # 9_1 SNAP coarse - 2x - 64m
EXPERIMENT_NAME="9_1_snap_coarse_64m_1mpp"
python -m maploc.train experiment.name=$EXPERIMENT_NAME \
        data.crop_size_meters=128 \
        data.max_init_error=96 \
        data.pixel_per_meter=1 \
        data.tiles_filename=tiles_1mpp.pkl \
        model.bev_mapper.z_max=64.0 \
        model.bev_mapper.x_max=64.0 \
        model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
        training.lr=5e-5

# # 9_2 SNAP coarse - 4x - 2mpp
# EXPERIMENT_NAME="9_2_snap_coarse_128m_2mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.crop_size_meters=256 \
#         data.max_init_error=192 \
#         data.pixel_per_meter=0.5 \
#         data.tiles_filename=tiles_2mpp.pkl \
#         model.bev_mapper.z_max=128.0 \
#         model.bev_mapper.x_max=128.0 \
#         model.bev_mapper.image_encoder.backbone.encoder=resnet18 \
#         training.lr=5e-5


exit 0
