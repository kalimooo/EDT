#! /bin/bash

#SBATCH -J SRx3
#SBATCH -o ./slurm_out/job-%j.out
#SBATCH --gres=gpu

source EDTvenv/bin/activate

python fine_tune.py --config ~/EDT/configs/SRx3_EDTB_ImageNet200K.py --model ~/EDT/pretrained/SRx3_EDTB_ImageNet200K.pth -d ~/EDT/data/Galenus/Galenus_EB_ ~/EDT/data/Galenus/Galenus_PB_ --output ~/EDT/fine_tuned_models/test --epochs 10 --lr 1e-4 --loss g_loss
# ~/EDT/data/training_data ~/EDT/data/Galenus/Galenus_EB_ ~/EDT/data/Galenus/Galenus_PB_