profiler_modes=(False True)

# # Exhaustive
# num_rotations=(64 128 256 512)
# crop_size_meters=(64 112 128)


# echo "===Starting Exhaustive Matching experiments>>>"
# for num_rot in "${num_rotations[@]}"
# do 
#     for csm in "${crop_size_meters[@]}"
#     do
#         for profiler_mode in "${profiler_modes[@]}"
#         do
#             command="python -m maploc.evaluation.mapillary --experiment OrienterNet_MGL \
#             model.num_rotations=$num_rot model.apply_map_prior=False data.scenes=["amsterdam","milan"] \
#             model.ransac_matcher=False model.profiler_mode=$profiler_mode \
#             model.use_map_cutout=True data.crop_size_meters=$csm"
#             echo "Running command: $command"
#             $command
#         done
#     done
# done


echo "<<<Finished Exhaustive Matching experiments"

echo "===Starting Ransac Matching experiments>>>"

# Ransac
num_pose_samples=(5000) #10000 20000 30000)
# num_pose_sampling_retries=(2 4 8 12)
crop_size_meters=(112) # 112 128)
ransac_grid_refinement=(True) #  False) # TODO: change grid boundaries

for nps in "${num_pose_samples[@]}"
do
    for csm in "${crop_size_meters}"
    do
        for gr in "${ransac_grid_refinement[@]}"
        do
            for profiler_mode in "${profiler_modes[@]}"
            do
                echo "gr: $gr"
                command="python -m maploc.evaluation.mapillary --experiment OrienterNet_MGL \
                model.apply_map_prior=False data.scenes=['amsterdam','milan'] \
                model.ransac_matcher=True model.profiler_mode=$profiler_mode \
                model.use_map_cutout=True data.crop_size_meters=$csm \
                model.num_pose_samples=$nps model.num_pose_sampling_retries=8 \
                model.ransac_grid_refinement=$gr"
                echo "Running command: $command"
                $command
            done
        done
    done
done
        


# Double check that the faster RANSAC doesn't harm perf
for profiler_mode in "${profiler_modes[@]}"
do
    command="python -m maploc.evaluation.mapillary --experiment OrienterNet_MGL \
            model.apply_map_prior=False data.scenes=["amsterdam","milan"] \
            model.ransac_matcher=True model.profiler_mode=$profiler_mode \
            model.use_map_cutout=True data.crop_size_meters=128 \
            model.num_pose_samples=20000 model.num_pose_sampling_retries=8 \
            model.ransac_grid_refinement=True model.compute_sim_relu_and_numvalid=True"
    $command
done