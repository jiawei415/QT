id=$1
cuda_id=0
group="2024$id"
logdir="/apdcephfs/share_1563664/ztjiaweixu/vdt_sz"
dataset_path="/apdcephfs/share_1563664/ztjiaweixu/datasets"

# breakout pong seaquest qbert
# medium expert mixed 
env=breakout
dataset=mixed

export CUDA_VISIBLE_DEVICES=$cuda_id

seed=0
for i in $(seq 2)
do
    tag=$(date "+%Y%m%d%H%M%S")
    python experiment_atari.py --env ${env} --dataset ${dataset} --seed ${seed} \
        --group ${group} --save_path ${logdir} \
        > ~/logs/${env}_${seed}_${tag}.out 2> ~/logs/${env}_${seed}_${tag}.err &
    echo "run $cuda_id $env $seed $tag"
    sleep 2.0
    let seed=$seed+1
done
let cuda_id=$cuda_id+1

python taiji/run_gpu.py

# ps -ef | grep rpto_torch | awk '{print $2}'| xargs kill -9
