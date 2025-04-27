import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhanceNet(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=256, num_heads=8, depth=6):
        super(EnhanceNet, self).__init__()
        self.entry = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(2)]
        )  # Fewer residual blocks

        # Vision Transformer module
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=64,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
        )

    def forward(self, x):
        x = F.relu(self.entry(x))
        x = self.res_blocks(x)
        x = self.vit(x)
        x = self.upsample(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, depth):
        super(VisionTransformer, self).__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            ),
            nn.Flatten(start_dim=2),  # Flatten spatial dimensions (H, W) into one
            nn.Unflatten(2, (-1, embed_dim)),  # Equivalent to rearranging
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Dynamically compute the number of patches
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)  # +1 for the class token
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=depth,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.patch_embed(x)  # Patch embedding: (B, embed_dim, H', W')
        x = x.flatten(2).transpose(
            1, 2
        )  # Flatten spatial dimensions and rearrange to (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(
            b, -1, -1
        )  # Expand class token to match batch size: (B, 1, embed_dim)
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # Concatenate class token: (B, num_patches + 1, embed_dim)

        # Dynamically adjust positional embeddings
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = F.interpolate(
                self.pos_embed[:, 1:].transpose(
                    1, 2
                ),  # Exclude class token and transpose to (B, embed_dim, num_patches)
                size=x.size(1) - 1,  # Match the number of patches
                mode="linear",
                align_corners=False,
            ).transpose(
                1, 2
            )  # Transpose back to (B, num_patches, embed_dim)
            pos_embed = torch.cat(
                [self.pos_embed[:, :1], pos_embed], dim=1
            )  # Add back the class token
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed  # Add positional embedding
        x = self.transformer(x)  # Transformer encoder
        x = self.norm(x)
        x = x[:, 1:]  # Remove class token

        # Reshape back to image
        num_patches = x.size(1)
        h = w = int(num_patches**0.5)  # Assuming square patches
        x = x.permute(0, 2, 1).reshape(b, -1, h, w)  # Reshape to (B, embed_dim, H', W')
        return x


def compute_snr(predicted, ground_truth):
    """
    Compute the Signal-to-Noise Ratio (SNR) between the predicted and ground truth images.

    Args:
        predicted (torch.Tensor): The predicted output from the model (B, C, H, W).
        ground_truth (torch.Tensor): The ground truth image (B, C, H, W).

    Returns:
        float: The SNR value in decibels (dB).
    """
    # Ensure the tensors are of the same shape
    assert (
        predicted.shape == ground_truth.shape
    ), "Predicted and ground truth must have the same shape."

    # Compute the signal power (mean squared value of the ground truth)
    signal_power = torch.mean(ground_truth**2)

    # Compute the noise power (mean squared error between predicted and ground truth)
    noise_power = torch.mean((predicted - ground_truth) ** 2)

    # Avoid division by zero
    if noise_power == 0:
        return float("inf")  # Infinite SNR if there is no noise

    # Compute SNR in decibels (dB)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()
