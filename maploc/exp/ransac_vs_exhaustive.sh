export KINETO_LOG_LEVEL=5
profiler_modes=(False True)

# Exhaustive
num_rotations=(64 128 256 512) # (64 128 256 512)
crop_size_meters=(64 80 90 112 128)

profiler_modes=(False)

echo "=== Starting Exhaustive Matching experiments >>>"
echo -e "\n\n"
for num_rot in "${num_rotations[@]}"
do 
    for csm in "${crop_size_meters[@]}"
    do
        # for profiler_mode in "${profiler_modes[@]}"
        # do
        # csm=64
        profiler_mode=False
        command="python -m maploc.evaluation.mapillary --experiment OrienterNet_MGL \
        model.num_rotations=$num_rot model.apply_map_prior=False data.scenes=["amsterdam","milan"] \
        model.ransac_matcher=False model.profiler_mode=$profiler_mode model.overall_profiler=True\
        model.use_map_cutout=True data.crop_size_meters=$csm"
        echo "Running command: $command"
        echo -e "\nStarting experiment with params: num_rot=$num_rot, csm=$csm, profiler_mode=$profiler_mode\n"
        $command
        echo -e "\nFinished experiment with params: num_rot=$num_rot, csm=$csm, profiler_mode=$profiler_mode\n"
        # done
    done
done

echo "<<<Finished Exhaustive Matching experiments"

echo "=== Starting Ransac Matching experiments >>>"

# Ransac
num_pose_samples=(5000 10000 20000 30000)
crop_size_meters=(64 80 90 112) # 128)

for nps in "${num_pose_samples[@]}"
do
    for csm in "${crop_size_meters[@]}"
    do
        for profiler_mode in "${profiler_modes[@]}"
        do
            gr=True
            command="python -m maploc.evaluation.mapillary --experiment OrienterNet_MGL \
            model.apply_map_prior=False data.scenes=['amsterdam','milan'] \
            model.ransac_matcher=True model.profiler_mode=$profiler_mode model.overall_profiler=False \
            model.use_map_cutout=True data.crop_size_meters=$csm \
            model.num_pose_samples=$nps model.num_pose_sampling_retries=8 \
            model.ransac_grid_refinement=$gr"
            echo "Running command: $command"
            echo -e "\nStarting experiment with params: nps=$nps, csm=$csm, gr=$gr, profiler_mode=$profiler_mode\n"
            $command
            echo -e "\nFinished experiment with params: nps=$nps, csm=$csm, gr=$gr, profiler_mode=$profiler_mode\n"
        done
    done
done
        

# Double check that the faster RANSAC doesn't harm perf
for profiler_mode in "${profiler_modes[@]}"
do
    command="python -m maploc.evaluation.mapillary --experiment OrienterNet_MGL \
            model.apply_map_prior=False data.scenes=["amsterdam","milan"] \
            model.ransac_matcher=True model.profiler_mode=$profiler_mode model.overall_profiler=False \
            model.use_map_cutout=True data.crop_size_meters=90 \
            model.num_pose_samples=20000 model.num_pose_sampling_retries=8 \
            model.ransac_grid_refinement=False model.compute_sim_relu_and_numvalid=True"
    $command
done