import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
import math
from torch.utils.data import DataLoader
from torchvision import transforms

# from div2k_dataset import DIV2KDataset  # Assuming this is implemented
from tqdm import tqdm
from PIL import Image
import os
import torch
from torch.utils.data import Dataset


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
        return hr_image  # Ensure this returns a single tensor, not a list


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


class EnhanceNetBase(nn.Module):
    """
    Base EnhanceNet model for super-resolution (without Vision Transformer)
    """

    def __init__(
        self, in_channels=3, out_channels=3, feature_channels=64, scale_factor=4
    ):
        super().__init__()

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Residual blocks for local feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(feature_channels) for _ in range(4)]  # 4 residual blocks
        )

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

    def _init_weights(self):
        # Initialize convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial feature extraction
        initial_features = self.initial(x)

        # Residual blocks
        res_features = self.residual_blocks(initial_features)

        # Upsampling
        upsampled = self.upsampler(res_features)

        # Final output
        output = self.final(upsampled).clamp(0, 1)  # Clamp output to [0, 1]
        return output


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
            *discriminator_block(128, 256, stride=1),
            *discriminator_block(256, 256, stride=2),
            *discriminator_block(256, 512, stride=1),
            *discriminator_block(512, 512, stride=2),
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten for fully connected layers
        x = self.fc(x)
        return x


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


class EnhanceNetTrainer:
    """
    Trainer for EnhanceNet model with perceptual and adversarial losses
    """

    def __init__(self, generator, discriminator, device="cuda"):
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        # Loss functions
        self.content_loss = nn.L1Loss().to(device)
        self.perceptual_loss = VGGPerceptualLoss().to(device)
        self.adversarial_loss = nn.MSELoss().to(device)  # Use least-squares loss

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


if __name__ == "__main__":
    # Create models
    generator = EnhanceNetBase(
        in_channels=3,
        out_channels=3,
        feature_channels=64,
        scale_factor=4,  # Scale factor for super-resolution
    )

    discriminator = Discriminator(in_channels=3)

    # Create trainer
    trainer = EnhanceNetTrainer(generator, discriminator, device="cuda")

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
