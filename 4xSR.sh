#! /bin/bash

#SBATCH -J SRx4
#SBATCH -o ./slurm_out/job-%j.out
#SBATCH --gres=gpu:L4:4

source EDTvenv/bin/activate

python fine_tune.py --config ~/EDT/configs/SRx4_EDTB_ImageNet200K.py --model ~/EDT/pretrained/SRx4_EDTB_ImageNet200K.pth -d ~/EDT/data/training_data --output ~/EDT/fine_tuned_models/test_not_sorted --epochs 10 --lr 1e-4 --loss g_loss
# ~/EDT/data/training_data ~/EDT/data/Galenus/Galenus_EB_ ~/EDT/data/Galenus/Galenus_PB_