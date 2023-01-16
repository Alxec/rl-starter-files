#!/usr/bin/env bash
#SBATCH --partition=unkillable
#SBATCH --mem=10GB
#SBATCH --time=12:00:00
#SBATCH --job-name=4rooms_vis_train_ac

source ~/PredictiveReplay/load_venv.sh

python3 -m  scripts.train --algo a2c --env MiniGrid-FourRooms-Fixed-v0 --wrapper FullyObsWrapper --model 4rooms__FO_visible_ac_13_01_23 --log-interval 4 --save-interval 50 --frames 10000000
