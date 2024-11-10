id="$1"
cuda_id=0
group="2024$id"
logdir="/apdcephfs/share_1563664/ztjiaweixu/vdt_sz"
dataset_path="/apdcephfs/share_1563664/ztjiaweixu/datasets"

task=pen # pen, hammer, door, relocate
dataset=expert # human, cloned, expert
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

seed=0

function run_experiment {
    local env="$1"
    local dataset="$2"
    local eta="$3"
    python experiment.py --seed $seed \
        --env "$env" --dataset "$dataset" \
        --eta "$eta" \
        --corruption_agent "$corruption_agent" \
        --corruption_seed "$corruption_seed" \
        --corruption_mode "$corruption_mode" \
        --corruption_obs "$corruption_obs" \
        --corruption_act "$corruption_act" \
        --corruption_rew "$corruption_rew" \
        --corruption_rew2 "$corruption_rew2" \
        --corruption_rate "$corruption_rate" \
        --corruption_step "$corruption_step" \
        --data_suffix "$data_suffix" \
        --down_sample "$down_sample" \
        --use_collected "$use_collected" \
        --group "$group" --dataset_path "$dataset_path" --save_path "$logdir" \
        > ~/logs/"${env}_${seed}_$(date "+%Y%m%d%H%M%S").out" 2> ~/logs/"${env}_${seed}_$(date "+%Y%m%d%H%M%S").err" &
}

for i in $(seq 4); do
    if [ "$corruption_mode" = "random" ]; then
        corruption_seed=$seed
    fi

    case "$dataset" in
        human)
            case "$task" in
                pen) run_experiment pen human 0.001 ;;
                hammer) run_experiment hammer human 0.1 ;;
                door) run_experiment door human 0.001 ;;
                relocate) run_experiment relocate human 0.001 ;;
            esac
            ;;
        cloned)
            case "$task" in
                pen) run_experiment pen cloned 0.1 ;;
                hammer) run_experiment hammer cloned 0.1 ;;
                door) run_experiment door cloned 0.001 ;;
                relocate) run_experiment relocate cloned 0.001 ;;
            esac
            ;;
        expert)
            case "$task" in
                pen) run_experiment pen expert 0.001 ;;
                hammer) run_experiment hammer expert 0.1 ;;
                door) run_experiment door expert 0.001 ;;
                relocate) run_experiment relocate expert 0.001 ;;
            esac
            ;;
    esac

    echo "run $cuda_id $task $seed"
    sleep 2.0
    seed=$((seed + 1))
done

cuda_id=$((cuda_id + 1))

python taiji/run_gpu.py