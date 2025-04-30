import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
import math
from torch.utils.data import DataLoader
from torchvision import transforms
from div2k_dataset import DIV2KDataset  # Assuming this is implemented
from tqdm import tqdm
from PIL import Image
import os
import torch
from torch.utils.data import Dataset


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
    """
    Residual block for the convolutional part of the network
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
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
        self.proj_back = nn.Linear(
            embed_dim, feature_channels * patch_size * patch_size
        )

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
        # Initial feature extraction
        initial_features = self.initial(x)

        # Residual blocks
        res_features = self.residual_blocks(initial_features)

        # Skip connection for later use
        skip_features = self.skip_conv(res_features)

        # Vision Transformer processing
        patches = self.patch_embed(res_features)

        # Dynamically adjust positional embeddings to match the number of patches
        if patches.size(1) != self.pos_embed.size(1):
            pos_embed = F.interpolate(
                self.pos_embed.transpose(
                    1, 2
                ),  # Transpose to (batch_size, embed_dim, num_patches)
                size=patches.size(1),  # Match the number of patches
                mode="linear",
                align_corners=False,
            ).transpose(
                1, 2
            )  # Transpose back to (batch_size, num_patches, embed_dim)
        else:
            pos_embed = self.pos_embed

        patches = patches + pos_embed
        transformer_features = self.transformer(patches)

        # Reshape transformer output back to spatial dimensions
        batch_size = transformer_features.shape[0]
        transformer_features = self.proj_back(transformer_features)
        transformer_features = transformer_features.reshape(
            batch_size, self.h_patches, self.w_patches, -1
        )
        transformer_features = transformer_features.permute(0, 3, 1, 2)

        # Combine with skip features
        transformer_features = self.transformer_proj(transformer_features)

        if transformer_features.shape[-2:] != skip_features.shape[-2:]:
            transformer_features = F.interpolate(
                transformer_features,
                size=skip_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        combined_features = transformer_features + skip_features

        # Upsampling
        upsampled = self.upsampler(combined_features)

        # Final output
        output = self.final(upsampled)

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

    def __init__(self, in_channels=3, input_size=512):
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
            *discriminator_block(128, 256, stride=1),
            *discriminator_block(256, 256, stride=2),
            *discriminator_block(256, 512, stride=1),
            *discriminator_block(512, 512, stride=2),
            nn.Flatten(),
        )

        # Dynamically calculate the flattened size
        flattened_size = self._get_flattened_size((in_channels, input_size, input_size))

        # Add Linear layers
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Create a dummy input
            output = self.model(
                dummy_input
            )  # Pass through all layers except the Linear layers
            return output.numel()  # Get the total number of elements

    def forward(self, x):
        x = self.model(x)
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
            self.generator.parameters(), lr=1e-4, betas=(0.9, 0.999)
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999)
        )

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

        g_loss = (
            self.content_weight * content_loss
            + self.perceptual_weight * perceptual_loss
            + self.adversarial_weight * adversarial_loss
        )

        g_loss.backward()
        self.generator_optimizer.step()

        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "content_loss": content_loss.item(),
            "perceptual_loss": perceptual_loss.item(),
            "adversarial_loss": adversarial_loss.item(),
        }

    def validate(self, val_dataloader):
        """
        Validate the model on validation data
        """
        self.generator.eval()
        total_psnr = 0
        total_samples = 0

        with torch.no_grad():
            for lr_images, hr_images in val_dataloader:
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


class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, transform=None):
        self.hr_dir = hr_dir
        self.image_files = sorted(
            [
                os.path.join(hr_dir, fname)
                for fname in os.listdir(hr_dir)
                if fname.endswith(".png") or fname.endswith(".jpg")
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            hr_image = self.transform(hr_image)
        return hr_image  # Return a single tensor, not a list


# Example usage
if __name__ == "__main__":
    # Create models
    generator = EnhanceNetViT(
        in_channels=3,
        out_channels=3,
        feature_channels=32,
        embed_dim=256,
        transformer_depth=6,
        num_heads=4,
        scale_factor=4,
        img_size=128,  # Update to match the low-resolution image size
    )

    discriminator = Discriminator(in_channels=3)

    # Create trainer
    trainer = EnhanceNetViTTrainer(generator, discriminator, device="cuda")

    # Dataset and DataLoader
    train_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),  # Resize HR images to 512x512
            transforms.ToTensor(),
        ]
    )

    train_dataset = DIV2KDataset(
        hr_dir="/home/kakudas/dev/jhu_705_643_final_project/research/EnhanceNet/data/DIV2K_train_HR",
        transform=train_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Training loop
    EPOCHS = 50
    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        generator.train()
        discriminator.train()

        epoch_losses = {
            "g_loss": 0.0,
            "d_loss": 0.0,
            "content_loss": 0.0,
            "perceptual_loss": 0.0,
            "adversarial_loss": 0.0,
        }

        for hr_images in tqdm(train_loader, desc="Training"):
            # Create low-resolution images (128x128) from high-resolution images (512x512)
            lr_images = F.interpolate(
                hr_images, size=(128, 128), mode="bicubic", align_corners=False
            )

            # Train one batch
            batch_losses = trainer.train_batch(lr_images, hr_images)

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += batch_losses[key]

        # Average losses for the epoch
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - "
            f"G Loss: {epoch_losses['g_loss']:.4f}, "
            f"D Loss: {epoch_losses['d_loss']:.4f}, "
            f"Content Loss: {epoch_losses['content_loss']:.4f}, "
            f"Perceptual Loss: {epoch_losses['perceptual_loss']:.4f}, "
            f"Adversarial Loss: {epoch_losses['adversarial_loss']:.4f}"
        )

        # Save models periodically
        if (epoch + 1) % 10 == 0:
            trainer.save_models(
                f"generator_epoch_{epoch + 1}.pth",
                f"discriminator_epoch_{epoch + 1}.pth",
            )

    # Save final models
    trainer.save_models("generator_final.pth", "discriminator_final.pth")
    print("Training completed. Models saved.")
