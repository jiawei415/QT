ratio=0.02
python ratio_dataset.py --env_name hopper-medium-v2 --ratio $ratio
python ratio_dataset.py --env_name walker2d-medium-v2 --ratio $ratio
python ratio_dataset.py --env_name halfcheetah-medium-v2 --ratio $ratio

ratio=0.1
python ratio_dataset.py --env_name hopper-medium-replay-v2 --ratio $ratio
python ratio_dataset.py --env_name walker2d-medium-replay-v2 --ratio $ratio
python ratio_dataset.py --env_name halfcheetah-medium-replay-v2 --ratio $ratio

ratio=0.01
python ratio_dataset.py --env_name hopper-medium-expert-v2 --ratio $ratio
python ratio_dataset.py --env_name walker2d-medium-expert-v2 --ratio $ratio
python ratio_dataset.py --env_name halfcheetah-medium-expert-v2 --ratio $ratio

# python ratio_dataset.py --env_name kitchen-complete-v0 --ratio $ratio
# python ratio_dataset.py --env_name kitchen-partial-v0 --ratio $ratio
# python ratio_dataset.py --env_name kitchen-mixed-v0 --ratio $ratio

# ratio=0.01
# python ratio_dataset.py --env_name door-expert-v0 --ratio $ratio
# python ratio_dataset.py --env_name pen-expert-v0 --ratio $ratio
# python ratio_dataset.py --env_name hammer-expert-v0 --ratio $ratio
# python ratio_dataset.py --env_name relocate-expert-v0 --ratio $ratio

# ratio=0.02
# python ratio_dataset.py --env_name antmaze-medium-diverse-v0 --ratio $ratio
# python ratio_dataset.py --env_name antmaze-medium-play-v0 --ratio $ratio
# python ratio_dataset.py --env_name antmaze-large-diverse-v0 --ratio $ratio
# python ratio_dataset.py --env_name antmaze-large-play-v0 --ratio $ratio


# 0.02
# hopper-medium-v2 2186 -> 43      999906 -> 19805
# walker2d-medium-v2 1190 -> 23    999995 -> 20147
# halfcheetah-medium-v2 1000 -> 20 1000000 -> 20000

# 0.1
# hopper-medium-replay-v2 2041 -> 204     402000 -> 32921
# walker2d-medium-replay-v2 1093 -> 109   302000 -> 27937
# halfcheetah-medium-replay-v2 202 -> 20  202000 -> 20000

# 0.01
# hopper-medium-expert-v2 3213 -> 32,     1999400 -> 19851
# walker2d-medium-expert-v2 2190 -> 21    1999209 -> 18762
# halfcheetah-medium-expert-v2 2000 -> 20 2000000 -> 20000

# kitchen-complete-v0
# kitchen-partial-v0
# kitchen-mixed-v0

# 0.01
# door-expert-v0 5000 -> 100 1000000 -> 20000
# pen-expert-v0 5000 -> 100 500000 -> 10000
# hammer-expert-v0 5000 -> 100 1000000 -> 20000
# relocate-expert-v0 5000 -> 100 1000000 -> 20000

# 0.02
# antmaze-medium-diverse-v0 2923 -> 58 999935 -> 27569
# antmaze-medium-play-v0 10694 -> 213 999260 -> 17681
# antmaze-large-diverse-v0 7140 -> 142 999618 -> 26369
# antmaze-large-play-v0 13457 -> 269 999540 -> 18125
