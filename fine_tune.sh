#! /bin/bash

#SBATCH --gres=gpu

source EDTvenv/bin/activate

python fine_tune.py --config ~/EDT/configs/SRx4_EDTB_ImageNet200K.py --model ~/EDT/pretrained/SRx4_EDTB_ImageNet200K.pth --data ~/EDT/data/training_data --output ~/EDT/fine_tuned_models/test --epochs 10 --lr 1e-4 --loss g_loss
