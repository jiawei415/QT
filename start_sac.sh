id=$1
cuda_id=0
group="2024$id"

logdir="/apdcephfs/share_1563664/ztjiaweixu/vdt_sz"

# halfcheetah hopper walker2d
for env in halfcheetah
do
    export CUDA_VISIBLE_DEVICES=$cuda_id

    seed=0
    for i in $(seq 4)
    do
        tag=$(date "+%Y%m%d%H%M%S")
        python run_sac.py --env ${env} --seed ${seed} \
        --group ${group} --save_path ${logdir} \
        > ~/logs/${env}_${seed}_${tag}.out 2> ~/logs/${env}_${seed}_${tag}.err &
        echo "run $cuda_id $env $seed  $tag"
        sleep 2.0
        let seed=$seed+1
    done
    let cuda_id=$cuda_id+1
done

python taiji/run_gpu.py

# ps -ef | grep rpto_torch | awk '{print $2}'| xargs kill -9
