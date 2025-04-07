#! /bin/bash

#SBATCH -J SRx3
#SBATCH -o ./slurm_out/job-%j.out
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu=16G
#SBATCH -p long
#SBATCH -w uranus

source EDTvenv/bin/activate

# DONT FORGET TO UPDATE MODEL!!
python fine_tune.py --config ~/EDT/configs/SRx3_EDTB_ImageNet200K.py --model ~/EDT/fine_tuned_models/first_try/fine_tuned_model_epoch6.pth -d ~/EDT/data/Galenus_EB ~/EDT/data/Eciton_FB ~/EDT/data/Galenus_PB ~/EDT/data/Eciton_PB  --output ~/EDT/fine_tuned_models/big_train --epochs 10 --lr 1e-4 --loss g_loss
# ~/EDT/data/training_data ~/EDT/data/Galenus/Galenus_EB_ ~/EDT/data/Galenus/Galenus_PB_