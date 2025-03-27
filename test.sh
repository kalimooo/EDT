#! /bin/bash

#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu=16G
#SBATCH -p short

source EDTvenv/bin/activate

python test.py