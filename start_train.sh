#!/bin/bash

id=$1
cuda_id=0
group="2024$id"
logdir="/apdcephfs/share_1563664/ztjiaweixu/vdt_sz"
dataset_path="/apdcephfs/share_1563664/ztjiaweixu/datasets"

task=hopper
corruption_agent=IQL
corruption_seed=2023
corruption_mode=none  # none, random, adversarial
corruption_obs=0.0
corruption_act=0.0
corruption_rew=0.0
corruption_rew2=0.0
corruption_rate=0.3
corruption_step=1
data_suffix="0_20000_none_collect"
down_sample=1
use_collected=0

declare -A eta_map
eta_map["halfcheetah"]="5.0"
eta_map["hopper"]="3.0"
eta_map["walker2d"]="2.0"
eta_map["complete"]="0.001"
eta_map["partial"]="0.01"
eta_map["mixed"]="0.01"

declare -A env_map
env_map["halfcheetah"]="halfcheetah"
env_map["hopper"]="hopper"
env_map["walker2d"]="walker2d"
env_map["complete"]="kitchen"
env_map["partial"]="kitchen"
env_map["mixed"]="kitchen"

declare -A dataset_map
dataset_map["halfcheetah"]="medium-replay"
dataset_map["hopper"]="medium-replay"
dataset_map["walker2d"]="medium-replay"
dataset_map["complete"]="complete"
dataset_map["partial"]="partial"
dataset_map["mixed"]="mixed"

seed=0
for i in $(seq 4); do
    if [ "$corruption_mode" = "random" ]; then
        corruption_seed=$seed
    fi
    
    tag=$(date "+%Y%m%d%H%M%S")
    env=${env_map[$task]}
    eta=${eta_map[$task]}
    dataset=${dataset_map[$task]}
    
    python experiment.py --seed $seed \
        --env $env --dataset $dataset \
        --eta $eta \
        --corruption_agent $corruption_agent \
        --corruption_seed $corruption_seed \
        --corruption_mode $corruption_mode \
        --corruption_obs $corruption_obs \
        --corruption_act $corruption_act \
        --corruption_rew $corruption_rew \
        --corruption_rew2 $corruption_rew2 \
        --corruption_rate $corruption_rate \
        --corruption_step $corruption_step \
        --data_suffix $data_suffix \
        --down_sample $down_sample \
        --use_collected $use_collected \
        --group $group --dataset_path $dataset_path --save_path $logdir \
        > ~/logs/${env}_${seed}_${tag}.out 2> ~/logs/${env}_${seed}_${tag}.err &
    
    echo "run $cuda_id $env $dataset $seed $tag"
    sleep 2.0
    let seed=$seed+1
done

let cuda_id=$cuda_id+1

python taiji/run_gpu.py