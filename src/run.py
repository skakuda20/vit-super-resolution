"""
Module: run.py

This script is used to train and test the EnhanceNet-ViT model for image super-resolution.
It includes the following functionalities:
    - Training the model with a dataset of high-resolution images.
    - Testing the model on a test dataset and visualizing the results.
    - Saving the best model weights based on validation loss.
    - Logging training and validation metrics to a text file.
    - Loading the model weights for testing.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


from DIV2KDataset import DIV2KDataset
from EnhanceNetViT import (
    EnhanceNetViT,
    EnhanceNetViTTrainer,
    Discriminator,
    test_enhancenet_gan,
)
from utils import load_config, get_project_root_path, log_message


if __name__ == "__main__":
    root_path = get_project_root_path()
   
   # Load configuration
    config = load_config("params.yaml")
    mode = config["mode"]
    in_channels = config["in_channels"]
    out_channels = config["out_channels"]
    feature_channels = config["feature_channels"]
    embed_dim = config["embed_dim"]
    transformer_depth = config["transformer_depth"]
    num_heads = config["num_heads"]
    scale_factor = config["scale_factor"]
    img_size = config["img_size"]

    model_weights = config["model_weights"]

    train_dataset = config["train_dataset"]
    validation_dataset = config["validation_dataset"]
 
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if mode == "train":
        generator = EnhanceNetViT(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            embed_dim=embed_dim,
            transformer_depth=transformer_depth,
            num_heads=num_heads,
            scale_factor=scale_factor,
            img_size=img_size,
        )

        discriminator = Discriminator(in_channels=3)

        # Create trainer
        trainer = EnhanceNetViTTrainer(generator, discriminator, device=device)

        # Dataset and DataLoader
        train_transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
            ]
        )

        train_dataset = DIV2KDataset(
            hr_dir=train_dataset,
            transform=train_transform,
        )

        train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)

        val_transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        val_dataset = DIV2KDataset(
            hr_dir=validation_dataset,
            transform=val_transform,
        )

        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Training loop
        epochs = config["epochs"]
        txt_log_file = Path(root_path, "enhancenet_vit_training_terminal_output.txt")

        best_loss = float("inf")
        patience = 10
        counter = 0

        for epoch in range(epochs):
            print(f"Epoch [{epoch + 1}/{epochs}]")
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
                f"Epoch [{epoch + 1}/{epochs}] - "
                f"G Loss: {epoch_losses['g_loss']:.4f}, "
                f"D Loss: {epoch_losses['d_loss']:.4f}, "
                f"Content Loss: {epoch_losses['content_loss']:.4f}, "
                f"Perceptual Loss: {epoch_losses['perceptual_loss']:.4f}, "
                f"Adversarial Loss: {epoch_losses['adversarial_loss']:.4f}"
            )

            log_message(
                txt_log_file,
                f"Epoch [{epoch + 1}/{epochs}] - G Loss: {epoch_losses['g_loss']:.4f}, D Loss: {epoch_losses['d_loss']:.4f}, Perceptual Loss: {epoch_losses['perceptual_loss']:.4f}",
            )

            # Check validation loss
            val_loss = trainer.validate(val_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                trainer.save_models("best_generator_vit.pth", "best_discriminator.pth")
            # else:
            #     counter += 1
            #     if counter >= patience:
            #         print("Early stopping triggered")
            #         break

        # Save the final model weights
        trainer.save_models("generator_final_vit.pth", "discriminator_final_vit.pth")
        print(
            "Final model weights saved: generator_final_vit.pth, discriminator_final.pth"
        )

    elif mode == "test":
        # Define test dataset and DataLoader
        test_transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),  # Resize HR images to 512x512
                transforms.ToTensor(),
            ]
        )
        test_dataset = DIV2KDataset(
            hr_dir=validation_dataset,
            transform=test_transform,
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Load trained generator model
        generator = EnhanceNetViT(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            embed_dim=embed_dim,
            transformer_depth=transformer_depth,
            num_heads=num_heads,
            scale_factor=scale_factor,
            img_size=img_size,
        )
        generator.load_state_dict(
            torch.load(
                model_weights,
                map_location=torch.device(device))
        )

        # Test the model
        test_enhancenet_gan(generator, test_loader, device=device, num_visualizations=5)
