#! /bin/bash

#SBATCH -J SRx4
#SBATCH -o ./slurm_out/job-%j.out
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu=16G
#SBATCH -p long
#SBATCH -w uranus

source EDTvenv/bin/activate

# DONT FORGET TO UPDATE MODEL!!
python fine_tune.py --config ~/EDT/configs/SRx4_EDTB_ImageNet200K --model ~/EDT/pretrained/SRx4_EDTB_ImageNet200K.pth -d ~/EDT/data/Eciton_FB --output ~/EDT/fine_tuned_models/big_train --epochs 10 --lr 1e-4 --loss g_loss
# ~/EDT/data/Galenus_PB ~/EDT/data/Eciton_PB ~/EDT/data/Galenus_FB