#!/bin/bash


#SBATCH -n 8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --ntasks-per-node=8
#SBATCH --account=ls_polle
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gres=gpumem:16G
#SBATCH --signal=INT@600

#SBATCH --job-name=$EXPERIMENT_NAME
#SBATCH --output=${EXPERIMENT_NAME}.out
#SBATCH --error=${EXPERIMENT_NAME}.err

# 0. OrienterNet fine
# EXPERIMENT_NAME="verification_orienternet_fine"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=True \
#         data.crop_size_meters=64 \
#         data.pixel_per_meter=2 \
#         data.tiles_filename=tiles.pkl \
#         model.bev_mapper.mode=forward \
#         model.bev_mapper.z_max=32.0 \
#         model.bev_mapper.x_max=32.0


# 1. SNAP fine
# EXPERIMENT_NAME="verification_snap_fine"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
        # data.rectify_image=False \
        # data.crop_size_meters=64 \
        # data.pixel_per_meter=2 \
        # data.tiles_filename=tiles.pkl \
        # model.bev_mapper.mode=inverse \
        # model.bev_mapper.z_max=32.0 \
        # model.bev_mapper.x_max=32.0 \


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
        # data.rectify_image=False \
        # data.crop_size_meters=256 \
        # data.pixel_per_meter=0.5 \
        # data.tiles_filename=tiles_2mpp.pkl \
        # model.bev_mapper.mode=inverse \
        # model.bev_mapper.z_max=128.0 \
        # model.bev_mapper.x_max=128.0


# 4. OrienterNet coarse - 2x - 1mpp
# EXPERIMENT_NAME="orienternet_coarse_64m_1mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
        # data.rectify_image=True \
        # data.crop_size_meters=128 \
        # data.pixel_per_meter=1 \
        # data.tiles_filename=tiles_1mpp.pkl \
        # model.bev_mapper.mode=forward \
        # model.bev_mapper.z_max=64.0 \
        # model.bev_mapper.x_max=64.0

# 5. SNAP coarse - 2x - 1mpp
# EXPERIMENT_NAME="snap_coarse_64m_1mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
        data.rectify_image=False \
        data.crop_size_meters=128 \
        data.pixel_per_meter=1 \
        data.tiles_filename=tiles_1mpp.pkl \
        model.bev_mapper.mode=inverse \
        model.bev_mapper.z_max=64.0 \
        model.bev_mapper.x_max=64.0



# ??? 6. OrienterNet coarse - 8x - 4mpp
# EXPERIMENT_NAME="orienternet_coarse_256m_4mpp"
# python -m maploc.train experiment.name=$EXPERIMENT_NAME \
#         data.rectify_image=True \
#         data.crop_size_meters=512 \
#         data.pixel_per_meter=1 \
#         data.tiles_filename=tiles_4mpp.pkl \
#         model.bev_mapper.mode=forward \
#         model.bev_mapper.z_max=256.0 \
#         model.bev_mapper.x_max=256.0