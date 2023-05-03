#!/usr/bin/env bash
#SBATCH --partition=unkillable
#SBATCH --mem=10GB
#SBATCH --time=12:00:00
#SBATCH --job-name=4rooms_vis_train_ac

source ~/PredictiveReplay/load_venv.sh

# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-v0 --pc True --sd 2 --model 21_03_23_Lava_3x4_PC_sd2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000

# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg025-v0 --pc True --model 20_03_23_Lava_3x4_PC_neg025 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000

python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --pc True --sd 8 --model 26_03_23_Lava_3x4_PC_neg024_1_sd8 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000 --seed 1
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --pc True --sd 8 --model 26_03_23_Lava_3x4_PC_neg024_2_sd8 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000 --seed 2
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --pc True --sd 8 --model 26_03_23_Lava_3x4_PC_neg024_3_sd8 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000 --seed 3
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --pc True --sd 8 --model 26_03_23_Lava_3x4_PC_neg024_4_sd8 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000 --seed 4
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --pc True --sd 8 --model 26_03_23_Lava_3x4_PC_neg024_5_sd8 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000 --seed 5

python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg027-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg027_1 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 1
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg027-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg027_2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 2
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg027-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg027_3 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 3
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg027-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg027_4 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 4
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg027-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg027_5 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 5

python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg03-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg03_1 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 1
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg03-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg03_2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 2
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg03-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg03_3 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 3
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg03-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg03_4 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 4
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg03-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg03_5 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 5

python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg035-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg035_1 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 1
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg035-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg035_2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 2
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg035-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg035_3 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 3
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg035-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg035_4 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 4
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg035-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg035_5 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 5

python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg04-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg04_1 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 1
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg04-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg04_2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 2
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg04-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg04_3 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 3
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg04-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg04_4 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 4
python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg04-v0 --pc True --model 26_03_23_Lava_3x4_PC_neg04_5 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 5

# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --pc True --sd 5 --model 23_03_23_Lava_3x4_PC_neg024_1 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 1
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --pc True --sd 5 --model 23_03_23_Lava_3x4_PC_neg024_2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 2
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --pc True --sd 5 --model 23_03_23_Lava_3x4_PC_neg024_3 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 3
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --pc True --sd 5 --model 23_03_23_Lava_3x4_PC_neg024_4 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 4
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --pc True --sd 5 --model 23_03_23_Lava_3x4_PC_neg024_5 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 5

python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --wrapper FullyObsWrapper --model 23_03_23_Lava_3x4_FO_neg024_1 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 1
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --wrapper FullyObsWrapper --model 23_03_23_Lava_3x4_FO_neg024_2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 2
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --wrapper FullyObsWrapper --model 23_03_23_Lava_3x4_FO_neg024_3 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 3
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --wrapper FullyObsWrapper --model 23_03_23_Lava_3x4_FO_neg024_4 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 4
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --wrapper FullyObsWrapper --model 23_03_23_Lava_3x4_FO_neg024_5 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 800000 --seed 5

# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --model 23_03_23_Lava_3x4_PO_neg024_6 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1600000 --seed 6
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --model 23_03_23_Lava_3x4_PO_neg024_7 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1600000 --seed 7
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --model 23_03_23_Lava_3x4_PO_neg024_8 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1600000 --seed 8
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --model 23_03_23_Lava_3x4_PO_neg024_9 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1600000 --seed 9
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg024-v0 --model 23_03_23_Lava_3x4_PO_neg024_10 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1600000 --seed 10

# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg023-v0 --model 22_03_23_Lava_3x4_PO_neg023_1 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1600000 --seed 1
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg023-v0 --model 22_03_23_Lava_3x4_PO_neg023_2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1600000 --seed 2
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg023-v0 --model 22_03_23_Lava_3x4_PO_neg023_3 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1600000 --seed 3
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg023-v0 --model 22_03_23_Lava_3x4_PO_neg023_4 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1600000 --seed 4
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg023-v0 --model 22_03_23_Lava_3x4_PO_neg023_5 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1600000 --seed 5
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg027-v0 --model 22_03_23_Lava_3x4_PO_neg027_3 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1600000 --seed 3

# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg035-v0 --pc True --model 20_03_23_Lava_3x4_PC_neg035 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000

# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-Target-5x5-4x4-v0 --wrapper FullyObsWrapper --model 02_02_23_Lava_target_4x4_FO_neg --seed 6 --save-interval 12 --discount 0.98 --frames 49152
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-v0 --wrapper FullyObsWrapper --model 03_02_23_Lava_target_4x4_FO_neg --seed 6 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-4x4-v0 --wrapper FullyObsWrapper --model 31_01_23_Lava_target_4x4_FO --save-interval 12 --discount 0.98 --lr 0.0001 --frames 1500000

# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg032-v0 --model 14_02_23_Lava_target_3x4_PO_neg032 --seed 2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 4000000
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg033-v0 --model 14_02_23_Lava_target_3x4_PO_neg033 --seed 2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 2000000
#python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-4x4-v0 --model 26_01_23_Lava_target_4x4_PO --save-interval 12 --discount 0.98 --lr 0.0001 --frames 1500000


# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-Target-5x5-4x4-v0 --wrapper FullyObsWrapper --model 01_02_23b_Lava_target_4x4_FO --seed 3 --save-interval 12 --discount 0.98 --frames 49152
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-4x4-v0 --wrapper FullyObsWrapper --model 01_02_23b_Lava_target_4x4_FO --seed 3 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000

# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-Target-5x5-3x4-neg04-v0 --model 03_02_23_Lava_target_3x4_PO_neg04 --seed 2 --save-interval 12 --discount 0.98 --frames 49152
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg034-v0 --model 14_02_23_Lava_target_3x4_PO_neg034 --seed 2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 2000000


# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-Target-5x5-4x4-v0 --wrapper FullyObsWrapper --model 01_02_23c_Lava_target_4x4_FO --seed 4 --save-interval 12 --discount 0.98 --frames 49152
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-4x4-v0 --wrapper FullyObsWrapper --model 01_02_23c_Lava_target_4x4_FO --seed 4 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000

# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-Target-5x5-3x4-neg06-v0 --model 03_02_23_Lava_target_3x4_PO_neg06 --seed 2 --save-interval 12 --discount 0.98 --frames 49152
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-3x4-neg031-v0 --model 14_02_23_Lava_target_3x4_PO_neg031 --seed 2 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 4000000


# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-Target-5x5-4x4-v0 --wrapper FullyObsWrapper --model 01_02_23d_Lava_target_4x4_FO --seed 5 --save-interval 12 --discount 0.98 --frames 49152
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-4x4-v0 --wrapper FullyObsWrapper --model 01_02_23d_Lava_target_4x4_FO --seed 5 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000

# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-Target-5x5-4x4-v0 --model 01_02_23d_Lava_target_4x4_PO --seed 5 --save-interval 12 --discount 0.98 --frames 49152
# python3 -m  scripts.train --algo ppo --env MiniGrid-FakeLava-5x5-4x4-v0 --model 01_02_23d_Lava_target_4x4_PO --seed 5 --save-interval 12 --discount 0.98 --lr 0.0007 --frames 1000000

python3 -m  scripts.visualize --env MiniGrid-FakeLava-5x5-4x4-v0 --model 01_02_23_Lava_target_4x4_PO_long