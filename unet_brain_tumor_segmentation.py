"""
U-Net Brain Tumor Segmentation (Binary)
========================================
A PyTorch implementation of U-Net for binary brain tumor segmentation
on the BraTS2020 2D Multichannel dataset.

Dataset: https://www.kaggle.com/datasets/alilebanizakaria/2d-multichannel-brats2020

Usage:
    python unet_brain_tumor_segmentation.py --data_dir "BraTS2020 2D Multichannel"
    python unet_brain_tumor_segmentation.py --data_dir "BraTS2020 2D Multichannel" --epochs 50 --batch_size 8
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# 1. U-NET ARCHITECTURE
# ============================================================================

class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net architecture for image segmentation.
    
    Args:
        in_channels: Number of input channels (3 for RGB).
        out_channels: Number of output channels (1 for binary segmentation).
        features: List of feature sizes for each encoder/decoder level.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (downsampling path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder (upsampling path)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder with skip connections
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            x = self.ups[i + 1](torch.cat((skip_connection, x), dim=1))

        return self.final_conv(x)


# ============================================================================
# 2. DATASET
# ============================================================================

class BraTSDataset(Dataset):
    """BraTS2020 2D dataset for binary tumor segmentation.
    
    Args:
        img_dir: Path to the directory containing MRI images.
        mask_dir: Path to the directory containing segmentation masks.
        file_pairs: List of (image_filename, mask_filename) tuples.
    """
    def __init__(self, img_dir, mask_dir, file_pairs):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_pairs = file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_filename, mask_filename = self.file_pairs[idx]

        # Load image as RGB and normalize to [0, 1]
        img = Image.open(os.path.join(self.img_dir, img_filename)).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0

        # Load mask as grayscale
        mask = Image.open(os.path.join(self.mask_dir, mask_filename)).convert('L')
        mask = np.array(mask).astype(np.float32)

        # Binary mask: map 255 (white) to 1.0 (tumor), clean up artifacts
        mask[mask == 255] = 1.0
        mask[mask > 1.0] = 0.0

        # Format for PyTorch: [C, H, W]
        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W] for binary

        return img, mask


# ============================================================================
# 3. METRICS
# ============================================================================

def dice_coeff_binary(preds, targets):
    """Compute binary Dice coefficient.
    
    Handles edge case where both prediction and target are empty
    (returns 1.0, since the model correctly predicted no tumor).
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = torch.where(
        union == 0,
        torch.tensor(1.0, device=preds.device),
        (2.0 * intersection) / (union + 1e-8)
    )
    return dice.mean().item()


# ============================================================================
# 4. DATA PREPARATION
# ============================================================================

def prepare_data(data_dir):
    """Load metadata and create train/val/test splits.
    
    Args:
        data_dir: Root directory containing Images/, Masks/, and metadata.csv
        
    Returns:
        Tuple of (img_dir, mask_dir, train_pairs, val_pairs, test_pairs)
    """
    img_dir = os.path.join(data_dir, 'Images')
    mask_dir = os.path.join(data_dir, 'Masks')
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    # Validate paths
    for path, name in [(img_dir, "Images"), (mask_dir, "Masks"), (metadata_path, "metadata.csv")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at: {path}")

    # Build valid image-mask pairs from metadata
    metadata_df = pd.read_csv(metadata_path)
    actual_img_files = set(os.listdir(img_dir))
    actual_mask_files = set(os.listdir(mask_dir))

    all_pairs = []
    for _, row in metadata_df.iterrows():
        img_f, mask_f = row['Image'], row['Mask']
        if img_f in actual_img_files and mask_f in actual_mask_files:
            all_pairs.append((img_f, mask_f))

    print(f"Valid image-mask pairs found: {len(all_pairs)}")

    # Split: 80% train, 10% val, 10% test
    train_pairs, temp_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)

    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    return img_dir, mask_dir, train_pairs, val_pairs, test_pairs


# ============================================================================
# 5. TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, save_path):
    """Train the U-Net model with validation and best model checkpointing.
    
    Returns:
        Tuple of (train_losses, val_dices) for plotting.
    """
    train_losses = []
    val_dices = []
    best_val_dice = 0.0

    print(f"\nStarting Binary Segmentation Training for {epochs} Epochs...")
    print(f"Device: {device}\n")

    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation phase ---
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                val_dice += dice_coeff_binary(model(images), masks)

        avg_val_dice = val_dice / len(val_loader)
        val_dices.append(avg_val_dice)

        print(f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        # Step scheduler based on validation dice
        scheduler.step(avg_val_dice)

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), save_path)
            print(f"   -> Best Model Saved! (Dice: {best_val_dice:.4f})")

    print(f"\nTraining Complete. Best Val Dice: {best_val_dice:.4f}")
    return train_losses, val_dices


# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def plot_learning_curves(train_losses, val_dices, epochs):
    """Plot training loss and validation dice curves."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, marker='o', markersize=2, color='blue', label='Train Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_dices, marker='o', markersize=2, color='orange', label='Validation Dice')
    plt.title('Validation Dice Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score (Higher is Better)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Learning curves saved to: learning_curves.png")


def visualize_prediction(model, test_loader, device):
    """Visualize a single test prediction alongside ground truth."""
    model.eval()
    sample_imgs, sample_masks = next(iter(test_loader))
    sample_img_gpu = sample_imgs[0].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_logit = model(sample_img_gpu)
        pred_binary = (torch.sigmoid(pred_logit) > 0.5).float()

    img_display = sample_imgs[0][0].numpy()
    true_mask_display = sample_masks[0][0].numpy()
    pred_mask_display = pred_binary[0][0].cpu().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_display, cmap='gray')
    plt.title("Original MRI (Test Image)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask_display, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask_display, cmap='gray')
    plt.title("U-Net Prediction")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('test_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Test prediction saved to: test_prediction.png")


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="U-Net Brain Tumor Segmentation (Binary)")
    parser.add_argument('--data_dir', type=str, default='BraTS2020 2D Multichannel',
                        help='Path to dataset directory containing Images/, Masks/, metadata.csv')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--pos_weight', type=float, default=5.0,
                        help='Positive class weight for BCE loss (handles class imbalance)')
    parser.add_argument('--model_path', type=str, default='best_unet_model.pth',
                        help='Path to save/load the best model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Data ---
    img_dir, mask_dir, train_pairs, val_pairs, test_pairs = prepare_data(args.data_dir)

    train_loader = DataLoader(BraTSDataset(img_dir, mask_dir, train_pairs),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(BraTSDataset(img_dir, mask_dir, val_pairs),
                            batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(BraTSDataset(img_dir, mask_dir, test_pairs),
                             batch_size=1, shuffle=False)

    # --- Model ---
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Train ---
    train_losses, val_dices = train_model(
        model, train_loader, val_loader, optimizer, scheduler,
        criterion, device, args.epochs, args.model_path
    )

    # --- Plot ---
    plot_learning_curves(train_losses, val_dices, args.epochs)

    # --- Test ---
    print("\n--- Running Final Test ---")
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()

    test_dice = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            test_dice += dice_coeff_binary(model(images), masks)

    avg_test_dice = test_dice / len(test_loader)
    print(f"FINAL TEST DICE SCORE: {avg_test_dice:.4f}")

    # --- Visualize ---
    visualize_prediction(model, test_loader, device)


if __name__ == "__main__":
    main()
