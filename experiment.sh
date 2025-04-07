#! /bin/bash

#SBATCH -J experiment
#SBATCH -p short
#SBATCH --gres=gpu:L4:1
#SBATCH --mem-per-gpu=16G
#SBATCH -o ./experiment_slurm_out/job-%j.out

source EDTvenv/bin/activate

python experiment.py --config ~/EDT/configs/SRx3_EDTB_ImageNet200K.py --model ~/EDT/pretrained/SRx3_EDTB_ImageNet200K.pth -d ~/EDT/data/small_set --output ~/EDT/fine_tuned_models/experiment --epochs 10 --lr 1e-4 --loss mse
