import argparse
import os
import cv2
import importlib
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

from utils.common import scandir
from utils.model_opr import load_model_filter_list
from utils.modules.edt import Network

# python fine_tune.py --config configs/SRx4_EDTB_ImageNet200K.py --model pretrained/SRx4_EDTB_ImageNet200K.pth --data data --output fine_tuned_output --epochs 10 --lr 1e-4

class GLoss(nn.Module):
    def __init__(self, scale_factor=4):
        """
        Initialize the G-Loss function
        Args:
            scale_factor: The super-resolution scale factor (2, 3, or 4)
        """
        super(GLoss, self).__init__()
        
        # Define the 8 gradient extraction kernels
        self.kernels = [
            [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],  # top-left
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],  # top
            [[0, 0, -1], [0, 1, 0], [0, 0, 0]],  # top-right
            [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],  # left
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],  # right
            [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],  # bottom-left
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],  # bottom
            [[0, 0, 0], [0, 1, 0], [0, 0, -1]]   # bottom-right
        ]
        
        # Convert the kernels to PyTorch tensors and register them as buffers
        self.grad_kernels = []
        for k in self.kernels:
            kernel = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0)
            self.register_buffer(f'kernel_{len(self.grad_kernels)}', kernel)
            self.grad_kernels.append(kernel)
            
        # Define the sampling rates based on scale factor
        self.scale_factor = scale_factor
        if scale_factor == 2:
            self.sampling_rates = [2]
        elif scale_factor == 3:
            self.sampling_rates = [2, 3]
        else:  # scale_factor == 4
            self.sampling_rates = [2, 4]

    def extract_gradients(self, x):
        """Extract gradient feature maps using 8-directional convolution kernels"""
        batch, channels = x.shape[0], x.shape[1]
        grad_maps = []
        
        # For each channel
        for c in range(channels):
            channel_grads = []
            # For each direction kernel
            for kernel_idx, kernel in enumerate(self.grad_kernels):
                # Apply convolution with padding
                kernel_var = getattr(self, f'kernel_{kernel_idx}')
                grad = F.conv2d(x[:, c:c+1], kernel_var.repeat(1, 1, 1, 1), padding=1)
                channel_grads.append(grad)
                
            # Stack the gradient maps from all directions
            channel_grad_maps = torch.cat(channel_grads, dim=1)
            grad_maps.append(channel_grad_maps)
            
        # Combine gradient maps from all channels
        return torch.cat(grad_maps, dim=1)

    def split_downsample(self, x, rate):
        """Downsample by splitting the image into blocks and stacking as channels"""
        b, c, h, w = x.size()
        # Reshape to create blocks of size rate x rate
        x = x.view(b, c, h // rate, rate, w // rate, rate)
        # Permute and reshape to convert spatial blocks to channels
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(b, c * (rate ** 2), h // rate, w // rate)
        return x

    def forward(self, sr, hr):
        """
        Calculate G-Loss between super-resolution and high-resolution images
        Args:
            sr: Super-resolution image tensor (B, C, H, W)
            hr: High-resolution ground truth tensor (B, C, H, W)
        """
        # 1. Calculate pixel loss (PL)
        pixel_loss = F.l1_loss(sr, hr)
        
        # 2. Calculate gradient loss (G1)
        sr_grads = self.extract_gradients(sr)
        hr_grads = self.extract_gradients(hr)
        g1_loss = F.l1_loss(sr_grads, hr_grads)
        
        total_loss = pixel_loss + g1_loss
        #total_loss = g1_loss
        
        # 3. Calculate split gradient losses (SGn)
        for rate in self.sampling_rates:
            # Downsample using splitting
            sr_downsampled = self.split_downsample(sr, rate)
            hr_downsampled = self.split_downsample(hr, rate)
            
            # Extract gradients from downsampled images
            sr_sg_grads = self.extract_gradients(sr_downsampled)
            hr_sg_grads = self.extract_gradients(hr_downsampled)
            
            # Calculate loss with weight adjustment (1/n²)
            weight = 1.0 / (rate ** 2)
            sg_loss = F.l1_loss(sr_sg_grads, hr_sg_grads) * weight
            
            total_loss = total_loss + sg_loss
        
        return total_loss

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

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SR model")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the config file, e.g., configs/SRx4_EDTB_ImageNet200K.py")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the pre-trained model, e.g., pretrained/SRx4_EDTB_ImageNet200K.pth")
    parser.add_argument('-d', '--data', nargs='+' ,type=str, required=True,
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

    # Load configuration module
    config_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    config_name = os.path.basename(args.config).split('.')[0]
    module_reg = importlib.import_module(f'configs.{config_name}')
    config = getattr(module_reg, 'Config', None)
    if config is None:
        raise ValueError("Config not found in the config file.")

    # Build model (using edt.py)
    model = Network(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    elif args.loss == "l2":
        criterion == nn.MSELoss()
    elif args.loss == 'ssim':
        criterion = ssim_loss
    else:  # 'g_loss'
        criterion = GLoss(scale_factor=args.scale).to(device)

    # Setup directories for LR and HR images.
    print(args.data)

    lr_files = []
    hr_files = []

    for path in args.data:
        lr_dir = os.path.join(path, "LR")
        hr_dir = os.path.join(path, "HR")
        lr_temp_list = []
        hr_temp_list = []

        if not os.path.isdir(lr_dir) or not os.path.isdir(hr_dir):
            raise ValueError(f"Both 'LR' and 'HR' subdirectories must exist under the data folder. It does not for {path}")    

        for file in os.listdir(lr_dir):
            if not os.path.isfile(os.path.join(lr_dir, file)):
                continue
            lr_temp_list.append(file[3:])
        
        for file in os.listdir(hr_dir):
            if not os.path.isfile(os.path.join(hr_dir, file)):
                continue
            hr_temp_list.append(file[3:])
        
        lr_files.append(lr_temp_list)
        hr_files.append(hr_temp_list)

    #lr_files = [f[3:] for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))]
    #hr_files = [f[3:] for f in os.listdir(hr_dir) if os.path.isfile(os.path.join(hr_dir, f))]

    # Only keep the common files which represent image pairs.

    common_files_list = []
    common_file_amount = 0

    for i in range(len(lr_files)):
        common_files = sorted(list(set(lr_files[i]).intersection(hr_files[i])))
        common_files_list.append(common_files)
        common_file_amount += len(common_files)

        if not common_files:
            raise ValueError("No matching files found between LR and HR for a directory.")

    print(f"Found {common_file_amount} image pairs for fine-tuning.")

    # Fine-tuning loop.
    average_loss_per_epoch = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0

        files_gone_through = 0

        for file_list in common_files_list:
            for filename in file_list:
                lr_path = os.path.join(lr_dir, "lr-" + filename)
                hr_path = os.path.join(hr_dir, "hr-" + filename)
                lr_img = read_image_to_tensor(lr_path).to(device)
                hr_img = read_image_to_tensor(hr_path).to(device)

                optimizer.zero_grad()
                # The network expects a list of tensors.
                output = model([lr_img])[0]

                # Check tensor range for debugging

                loss = criterion(output, hr_img)
                loss.backward()
                optimizer.step()

                files_gone_through += 1
                if (files_gone_through % 100) == 0:
                    print(f"Trained on {files_gone_through} images in epoch {epoch}")
                
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(common_files)
        average_loss_per_epoch.append(avg_loss)
        print(f"Epoch [{epoch}/{args.epochs}] - Average Loss: {avg_loss:.4f}")
        print(f"Average loss per epoch: {average_loss_per_epoch}")

    # Save the fine-tuned model.
    os.makedirs(args.output, exist_ok=True)
    save_path = os.path.join(args.output, "fine_tuned_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved at: {save_path}")

if __name__ == "__main__":
    main()