import logging
import argparse
import os
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim
import cv2
import numpy as np
import time

from utils.modules.edt import Network
from utils.model_opr import load_model_filter_list
from torch.utils.data import Dataset, DataLoader

def ssim_loss(pred, gt):
    return 1 - ssim(pred, gt, data_range=1.0)

def read_image_to_tensor(ipath):
    # NOTE this might not work? idk good to keep in mind
    img = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
    img = np.stack([img, img, img], axis=-1) # shape becomes (H, W, 3)

    # Convert BGR to RGB, transpose to (C,H,W), normalize and add batch dimension.
    img = np.transpose(img[:, :, ::-1], (2, 0, 1)).astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0)
    return img

class ImageDataSet(Dataset):

    def __init__(self, length, data):
        self.len = length
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename="experiment.log", encoding="utf-8", level=logging.INFO)

    logger.info("-----------------------------------------")
    logger.info("---------- STARTING EXPERIMENT ----------")
    logger.info("-----------------------------------------")

    parser = argparse.ArgumentParser(description="Experimenting for SR model")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the config file, e.g., configs/SRx4_EDTB_ImageNet200K.py")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the pre-trained model, e.g., pretrained/SRx4_EDTB_ImageNet200K.pth")
    parser.add_argument('-d', '--data', type=str, required=True,
                        help="Path to folder(s) containing 'LR' and 'HR' subdirectories")
    parser.add_argument('--output', type=str, default="output",
                        help="Directory to save the fine-tuned model")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of fine-tuning epochs")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--scale', type=int, default=4,
                        help="Super resolution scale factor (2, 3 or 4)")
    parser.add_argument('--loss', type=str, default='ssim', choices=['l1', 'mse', 'ssim', 'g_loss'],
                        help="Loss function to use (l1, mse, ssim or g_loss)")
    args = parser.parse_args()

    # Load configuration moduleEDT
    config_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    config_name = os.path.basename(args.config).split('.')[0]
    module_reg = importlib.import_module(f'configs.{config_name}')
    config = getattr(module_reg, 'Config', None)
    if config is None:
        raise ValueError("Config not found in the config file.")
    
    # Build model (using edt.py)
    model = Network(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = nn.DataParallel(model)
    logger.info(f"We have {torch.cuda.device_count()} GPUs")


    model = model.to(device)

    # Load pre-trained weights.
    if torch.cuda.is_available():
        load_model_filter_list(model, args.model, filter_list=[], cpu=False)
    else:
        load_model_filter_list(model, args.model, filter_list=[], cpu=True)

    # Prepare optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.loss == 'l1':
        criterion = nn.L1Loss()
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == 'ssim':
        criterion = ssim_loss

    # Setup directories for LR and HR images.
    lr_dir = os.path.join(args.data, "LR")
    hr_dir = os.path.join(args.data, "HR")
    if not os.path.isdir(lr_dir) or not os.path.isdir(hr_dir):
        raise ValueError("Both 'LR' and 'HR' subdirectories must exist under the data folder.")

    lr_files = [f[3:] for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))]
    hr_files = [f[3:] for f in os.listdir(hr_dir) if os.path.isfile(os.path.join(hr_dir, f))]

    # Only keep the common files which represent image pairs.
    common_files = list(set(lr_files).intersection(hr_files))
    if not common_files:
        raise ValueError("No matching files found between LR and HR directories.")
    logger.info(f"Found {len(common_files)} image pairs for fine-tuning.")

    before_loading = time.time()

    lr_images = []
    hr_images = []

    for filename in common_files:
        lr_path = os.path.join(lr_dir, "lr-" + filename)
        hr_path = os.path.join(hr_dir, "hr-" + filename)
        lr_images.append(read_image_to_tensor(lr_path).to(device))
        hr_images.append(read_image_to_tensor(hr_path).to(device))

    #lr_loader = DataLoader(dataset=ImageDataSet(len(lr_images), lr_images), shuffle=False)
    #hr_loader = DataLoader(dataset=ImageDataSet(len(hr_images), hr_images), shuffle=False)

    after_loading = time.time()

    logger.info(f"Took {after_loading - before_loading} to load all files to tensor")

    # Fine-tuning loop.
    average_loss_per_epoch = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        files_gone_through = 0
        time_inferencing = 0

        # for data in lr_loader:
        #     lr_image = lr_images[0]
        #     input = data.to(device)

        #     #print(lr_image)
        #     #print(input[0])

        #     before_inference = time.time()
        #     output = model([input[0]])[0]
        #     after_inference = time.time()

        #     time_inferencing += after_inference - before_inference

        #     files_gone_through += 1

        #     if files_gone_through % 100 == 0:
        #         logger.info(f"Trained on {files_gone_through} files in epoch {epoch}.")
        #         #logger.info(f"Average time loading a file over last 100 files: {time_loading / 100}")
        #         logger.info(f"Average time inferencing an image over last 100 images: {time_inferencing / 100}")
        #         time_loading = 0
        #         time_inferencing = 0

        for i in range(len(common_files)):
            lr_img = lr_images[i]
            hr_img = hr_images[i]

            optimizer.zero_grad()
            
            before_inference = time.time()
            output = model([lr_img])[0]
            after_inference = time.time()

            files_gone_through += 1

            loss = criterion(output, hr_img)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            #time_loading += after_loading - before_loading
            time_inferencing += after_inference - before_inference

            if files_gone_through % 100 == 0:
                logger.info(f"Trained on {files_gone_through} files in epoch {epoch}.")
                #logger.info(f"Average time loading a file over last 100 files: {time_loading / 100}")
                logger.info(f"Average time inferencing an image over last 100 images: {time_inferencing / 100}")
                time_loading = 0
                time_inferencing = 0

        avg_loss = epoch_loss / len(common_files)
        average_loss_per_epoch.append(avg_loss)
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Average Loss: {avg_loss:.4f}")
        logger.info(f"Average loss per epoch: {average_loss_per_epoch}")

    # Save the fine-tuned model.
    os.makedirs(args.output, exist_ok=True)
    save_path = os.path.join(args.output, "fine_tuned_model.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Fine-tuned model saved at: {save_path}")


if __name__ == "__main__":
    main()