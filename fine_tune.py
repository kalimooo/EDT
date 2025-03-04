import argparse
import os
import cv2
import importlib
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from utils.common import scandir
from utils.model_opr import load_model_filter_list
from utils.modules.edt import Network

# python fine_tune.py --config configs/SRx4_EDTB_ImageNet200K.py --model pretrained/SRx4_EDTB_ImageNet200K.pth --data data --output fine_tuned_output --epochs 10 --lr 1e-4

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
    parser.add_argument('--data', type=str, required=True,
                        help="Path to folder containing 'LR' and 'HR' subdirectories")
    parser.add_argument('--output', type=str, default="output",
                        help="Directory to save the fine-tuned model")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of fine-tuning epochs")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
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

    # Load pre-trained weights. For SR x4, we use the filter ['x4'].
    load_model_filter_list(model, args.model, filter_list=['x4'], cpu=True)

    # Prepare optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Setup directories for LR and HR images.
    lr_dir = os.path.join(args.data, "LR")
    hr_dir = os.path.join(args.data, "HR")
    if not os.path.isdir(lr_dir) or not os.path.isdir(hr_dir):
        raise ValueError("Both 'LR' and 'HR' subdirectories must exist under the data folder.")

    lr_files = [f for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))]
    hr_files = [f for f in os.listdir(hr_dir) if os.path.isfile(os.path.join(hr_dir, f))]

    # Only keep the common files which represent image pairs.
    common_files = sorted(list(set(lr_files).intersection(hr_files)))
    if not common_files:
        raise ValueError("No matching files found between LR and HR directories.")
    print(f"Found {len(common_files)} image pairs for fine-tuning.")

    # Fine-tuning loop.
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for filename in common_files:
            lr_path = os.path.join(lr_dir, filename)
            hr_path = os.path.join(hr_dir, filename)
            lr_img = read_image_to_tensor(lr_path).to(device)
            hr_img = read_image_to_tensor(hr_path).to(device)

            optimizer.zero_grad()
            # The network expects a list of tensors.
            output = model([lr_img])[0]
            loss = criterion(output, hr_img)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(common_files)
        print(f"Epoch [{epoch}/{args.epochs}] - Average Loss: {avg_loss:.4f}")

    # Save the fine-tuned model.
    os.makedirs(args.output, exist_ok=True)
    save_path = os.path.join(args.output, "fine_tuned_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved at: {save_path}")

if __name__ == "__main__":
    main()