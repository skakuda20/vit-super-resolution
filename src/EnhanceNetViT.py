"""
EnhanceNet-ViT: A Vision Transformer-based Model for Image Super-Resolution
This code implements a super-resolution model using a combination of convolutional layers and Vision Transformer (ViT) architecture.
The model is designed to enhance low-resolution images by learning high-frequency details and textures.

The model consists of several components:
    - PatchEmbedding: Converts input images into patch embeddings for the ViT.
    - TransformerEncoder: Implements the transformer encoder with multi-head self-attention.
    - ResidualBlock: A residual block for local feature extraction.
    - PixelShuffleUpsampler: Upsampling module using pixel shuffle.
    - EnhanceNetViT: The main model that combines convolutional layers and ViT for super-resolution.
    - VGGPerceptualLoss: A perceptual loss function using VGG19 features.
    - Discriminator: A discriminator network for adversarial training.
    - EnhanceNetViTTrainer: A trainer class for training the model with perceptual and adversarial losses.

The model is trained using a combination of content loss, perceptual loss, and adversarial loss to generate high-quality super-resolved images.
The code also includes a trainer class that handles the training process, including the optimization of both the generator and discriminator networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
import math
from torch.utils.data import DataLoader
from torchvision import transforms


import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from evaluation_metrics import calculate_lpips

from DIV2KDataset import DIV2KDataset


class PatchEmbedding(nn.Module):
    """
    Convert input images into patch embeddings for Vision Transformer
    """

    def __init__(self, img_size=48, patch_size=4, in_channels=64, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # (B, embed_dim, H', W') -> (B, embed_dim, n_patches)
        x = x.flatten(2)
        # (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multi-head self-attention
    """

    def __init__(self, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with multi-head self-attention and MLP
    """

    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention block
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        # MLP block
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += residual
        return out


class PixelShuffleUpsampler(nn.Module):
    """
    Upsampling module using pixel shuffle
    """

    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * (scale_factor**2), kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


class EnhanceNetViT(nn.Module):
    """
    Combined EnhanceNet and Vision Transformer model for super-resolution
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        feature_channels=64,
        embed_dim=256,
        transformer_depth=6,
        num_heads=8,
        scale_factor=4,
        img_size=48,
    ):
        super().__init__()

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Residual blocks for local feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(feature_channels) for _ in range(4)]  # Reduce from 8 to 4
        )

        # Vision Transformer components
        patch_size = 4
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=feature_channels,
            embed_dim=embed_dim,
        )

        # Add positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (img_size // patch_size) ** 2, embed_dim)
        )

        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            depth=3,
            num_heads=num_heads,  # Reduce depth from 6 to 3
        )

        # Reshape transformer output back to spatial features
        self.h_patches = self.w_patches = img_size // patch_size

        # Projection from transformer features back to conv features
        self.proj_back = nn.Linear(embed_dim, feature_channels)

        # Skip connection from early features
        self.skip_conv = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

        # Upsampling layers
        self.upsampler = nn.Sequential()
        log_scale_factor = int(math.log2(scale_factor))
        for _ in range(log_scale_factor):
            self.upsampler.add_module(
                f"upsample_{_}", PixelShuffleUpsampler(feature_channels, scale_factor=2)
            )

        # Final output layer
        self.final = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1)

        # Initialize weights
        self._init_weights()

        # Add this in the __init__ method of EnhanceNetViT
        self.transformer_proj = nn.Conv2d(
            in_channels=feature_channels
            * patch_size
            * patch_size,  # Match the output of proj_back
            out_channels=feature_channels,  # Project back to feature_channels
            kernel_size=1,
        )

    def _init_weights(self):
        # Initialize transformer weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")

        # Initial feature extraction
        initial_features = self.initial(x)
        # print(f"Initial features shape: {initial_features.shape}")

        # Residual blocks
        res_features = self.residual_blocks(initial_features)
        # print(f"Residual features shape: {res_features.shape}")

        # Skip connection
        skip_features = self.skip_conv(res_features)
        # print(f"Skip features shape: {skip_features.shape}")

        # Patch embedding
        patches = self.patch_embed(res_features)
        # print(f"Patches shape: {patches.shape}")
        # print(f"Positional embedding shape: {self.pos_embed.shape}")

        # Adjust positional embedding to match the number of patches
        if patches.size(1) != self.pos_embed.size(1):
            pos_embed = F.interpolate(
                self.pos_embed.transpose(
                    1, 2
                ),  # (1, num_patches, embed_dim) -> (1, embed_dim, num_patches)
                size=patches.size(1),  # Match the number of patches
                mode="linear",
                align_corners=False,
            ).transpose(
                1, 2
            )  # (1, embed_dim, num_patches) -> (1, num_patches, embed_dim)
        else:
            pos_embed = self.pos_embed

        # Add positional embedding
        patches = patches + pos_embed[:, : patches.size(1), :]
        transformer_features = self.transformer(patches)
        # print(f"Transformer features shape: {transformer_features.shape}")

        # Reshape transformer output
        batch_size = transformer_features.shape[0]
        num_patches = transformer_features.shape[1]
        embed_dim = transformer_features.shape[2]
        h_patches = w_patches = int(math.sqrt(num_patches))

        # Project transformer features back to spatial dimensions
        transformer_features = self.proj_back(transformer_features)
        feature_channels = self.patch_embed.patch_size**2
        transformer_features = transformer_features.reshape(
            batch_size,
            -1,
            h_patches,  # This should be 64 instead of 128
            w_patches,  # This should be 64 instead of 128
        )
        # print(f"Reshaped transformer features shape: {transformer_features.shape}")

        # Combine with skip features
        # print(f"Skip features shape: {skip_features.shape}")
        if transformer_features.shape[-2:] != skip_features.shape[-2:]:
            transformer_features = F.interpolate(
                transformer_features,
                size=skip_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        combined_features = transformer_features + skip_features
        # print(f"Combined features shape: {combined_features.shape}")

        # Upsampling
        upsampled = self.upsampler(combined_features)
        # print(f"Upsampled features shape: {upsampled.shape}")

        # Final output
        output = self.final(upsampled)
        # print(f"Final output shape: {output.shape}")

        return output


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features
    """

    def __init__(self, feature_layers=[2, 7, 12, 21]):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.feature_layers = feature_layers
        self.slice_indices = [0]

        for i in range(max(feature_layers) + 1):
            if i in feature_layers:
                self.slice_indices.append(i + 1)

        self.slices = nn.ModuleList()
        for i in range(len(self.slice_indices) - 1):
            start, end = self.slice_indices[i], self.slice_indices[i + 1]
            self.slices.append(nn.Sequential(*list(vgg[start:end])))

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x, target):
        x_features = self._get_features(x)
        target_features = self._get_features(target)

        loss = 0.0
        for x_feat, target_feat in zip(x_features, target_features):
            loss += F.mse_loss(x_feat, target_feat)

        return loss

    def _get_features(self, x):
        features = []
        for slice in self.slices:
            x = slice(x)
            features.append(x)
        return features


class Discriminator(nn.Module):
    """
    Discriminator network for adversarial training
    """

    def __init__(self, in_channels=3):
        super().__init__()

        def discriminator_block(in_channels, out_channels, stride=1, batch_norm=True):
            layers = [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=stride, padding=1
                )
            ]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, stride=1, batch_norm=False),
            *discriminator_block(64, 64, stride=2),
            *discriminator_block(64, 128, stride=1),
            *discriminator_block(128, 128, stride=2),
            *discriminator_block(128, 256, stride=1),  # Removed deeper blocks
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 512),  # Adjust input size to match the reduced model
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten for fully connected layers
        x = self.fc(x)
        return x


# Training utilities
class EnhanceNetViTTrainer:
    """
    Trainer for EnhanceNet-ViT model with perceptual and adversarial losses
    """

    def __init__(self, generator, discriminator, device="cuda"):
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        # Loss functions
        self.content_loss = nn.L1Loss().to(device)
        self.perceptual_loss = VGGPerceptualLoss().to(device)
        self.adversarial_loss = nn.BCEWithLogitsLoss().to(device)

        # Optimizers
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-6, betas=(0.5, 0.999)
        )

        # TODO: add learning rate scheduler

        # Loss weights
        self.content_weight = 1.0
        self.perceptual_weight = 0.1
        self.adversarial_weight = 0.001

    def train_batch(self, lr_images, hr_images):
        """
        Train one batch of data

        Args:
            lr_images: Low-resolution images (B, C, H, W)
            hr_images: High-resolution ground truth images (B, C, 4*H, 4*W)
        """
        lr_images = lr_images.to(self.device)
        hr_images = hr_images.to(self.device)

        # Generate SR images
        sr_images = self.generator(lr_images)

        # Train discriminator
        self.discriminator_optimizer.zero_grad()

        real_preds = self.discriminator(hr_images)
        fake_preds = self.discriminator(sr_images.detach())

        real_labels = torch.ones_like(real_preds)
        fake_labels = torch.zeros_like(fake_preds)

        d_real_loss = self.adversarial_loss(real_preds, real_labels)
        d_fake_loss = self.adversarial_loss(fake_preds, fake_labels)
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        self.discriminator_optimizer.step()

        # Train generator
        self.generator_optimizer.zero_grad()

        # Recalculate fake predictions for generator
        fake_preds = self.discriminator(sr_images)

        # Calculate losses
        content_loss = self.content_loss(sr_images, hr_images)
        perceptual_loss = self.perceptual_loss(sr_images, hr_images)
        adversarial_loss = self.adversarial_loss(fake_preds, real_labels)
        color_loss = F.l1_loss(sr_images, hr_images)  # Add color loss

        g_loss = (
            self.content_weight * content_loss
            + self.perceptual_weight * perceptual_loss
            + self.adversarial_weight * adversarial_loss
            + 0.1 * color_loss  # Add color loss with a small weight
        )

        g_loss.backward()
        self.generator_optimizer.step()

        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "content_loss": content_loss.item(),
            "perceptual_loss": perceptual_loss.item(),
            "adversarial_loss": adversarial_loss.item(),
            "color_loss": color_loss.item(),  # Track color loss
        }

    def validate(self, val_dataloader):
        """
        Validate the model on validation data
        """
        self.generator.eval()
        total_psnr = 0
        total_samples = 0

        with torch.no_grad():
            for hr_images in val_dataloader:
                lr_images = F.interpolate(
                    hr_images, size=(128, 128), mode="bicubic", align_corners=False
                )
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)

                sr_images = self.generator(lr_images)

                # Calculate PSNR
                mse = F.mse_loss(sr_images, hr_images)
                psnr = 10 * torch.log10(1.0 / mse)

                total_psnr += psnr.item() * lr_images.size(0)
                total_samples += lr_images.size(0)

        self.generator.train()
        return total_psnr / total_samples

    def save_models(self, generator_path, discriminator_path):
        """
        Save model weights
        """
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)

    def load_models(self, generator_path, discriminator_path):
        """
        Load model weights
        """
        self.generator.load_state_dict(torch.load(generator_path))
        self.discriminator.load_state_dict(torch.load(discriminator_path))


def test_enhancenet_gan(generator, test_loader, device="cuda", num_visualizations=5):
    """
    Test the EnhanceNet GAN model and visualize results.

    Args:
        generator (nn.Module): The trained EnhanceNet generator model.
        test_loader (DataLoader): DataLoader for the test dataset (provides only HR images).
        device (str): Device to run the model on ("cuda" or "cpu").
        num_visualizations (int): Number of images to visualize.
    """
    generator.eval()  # Set the generator to evaluation mode
    generator.to(device)

    total_psnr_sr = 0
    total_psnr_lr = 0
    total_ssim_sr = 0
    total_samples = 0
    visualized = 0

    with torch.no_grad():
        for hr_images in tqdm(test_loader, desc="Testing"):
            hr_images = hr_images.to(device)

            # Create low-resolution images dynamically
            lr_images = F.interpolate(
                hr_images, size=(128, 128), mode="bicubic", align_corners=False
            )

            # Generate super-resolution images
            sr_images = generator(lr_images)

            # Upsample LR images back to HR size for comparison
            lr_upsampled = F.interpolate(
                lr_images,
                size=hr_images.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )

            # Calculate PSNR for SR images
            mse_sr = F.mse_loss(sr_images, hr_images)
            psnr_sr = 10 * torch.log10(1.0 / mse_sr)
            total_psnr_sr += psnr_sr.item() * hr_images.size(0)

            # Calculate PSNR for LR images (upsampled)
            mse_lr = F.mse_loss(lr_upsampled, hr_images)
            psnr_lr = 10 * torch.log10(1.0 / mse_lr)
            total_psnr_lr += psnr_lr.item() * hr_images.size(0)

            total_samples += hr_images.size(0)

            # Calculate SSIM for SR images
            sr_image_np = sr_images[0].cpu().numpy().transpose(1, 2, 0)
            hr_image_np = hr_images[0].cpu().numpy().transpose(1, 2, 0)
            ssim_score = ssim(
                sr_image_np, hr_image_np, multichannel=True, win_size=3, data_range=1.0
            )  # Specify data_range=1.0
            total_ssim_sr += ssim_score
            # print(f"SSIM: {ssim_score:.4f}")

            # Visualize results
            if visualized < num_visualizations:

                lr_image = TF.to_pil_image(lr_images[0].cpu().clamp(0, 1))
                hr_image = TF.to_pil_image(hr_images[0].cpu().clamp(0, 1))
                sr_image = TF.to_pil_image(sr_images[0].cpu().clamp(0, 1))

                upscaled_lr = F.interpolate(
                    lr_images[0].cpu().unsqueeze(0),
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False,
                )

                lr_lpips_score = calculate_lpips(upscaled_lr, hr_images[0].cpu())
                sr_lpips_score = calculate_lpips(sr_images[0].cpu(), hr_images[0].cpu())
                print(f"LPIPS: {lr_lpips_score:.4f} --> {sr_lpips_score:.4f}")
                print(f"LPIPS: {sr_lpips_score:.4f}")
                print(f"SSIM: {ssim_score:.4f}")
                print(f"PSNR (SR): {psnr_sr:.2f} dB")
                print(psnr_lr, "->", psnr_sr)

                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.title(
                    f"High-Resolution Ground Truth\nSize: ({hr_image.size[0]}x{hr_image.size[1]})"
                )
                plt.imshow(hr_image)
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.title(
                    f"Low-Resolution Input\nSize: ({lr_image.size[0]}x{lr_image.size[1]})"
                )
                plt.imshow(lr_image)
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.title(
                    f"Super-Resolution Output\nSize: ({sr_image.size[0]}x{sr_image.size[1]})"
                )
                plt.imshow(sr_image)
                plt.axis("off")

                plt.show()
                visualized += 1

    # Calculate average PSNR and SSIM
    avg_psnr_sr = total_psnr_sr / total_samples if total_samples > 0 else 0.0
    avg_psnr_lr = total_psnr_lr / total_samples if total_samples > 0 else 0.0
    avg_ssim_sr = total_ssim_sr / total_samples if total_samples > 0 else 0.0
    print(f"Average PSNR (Super-Resolution): {avg_psnr_sr:.2f} dB")
    print(f"Average PSNR (Low-Resolution Upsampled): {avg_psnr_lr:.2f} dB")
    print(f"Average SSIM (Super-Resolution): {avg_ssim_sr:.4f}")
