import os
import time
from datetime import timedelta
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_processing.eeg_dataset import EEGGanDataset
from dataset_processing.gan.discriminator import Discriminator
from dataset_processing.gan.generator import Generator


class EEGGAN:
    def __init__(
            self,
            dataframe, generator_initial_layers: List[int], discriminator_layers: List[int],
            # For DataLoader
            batch_size=10, num_workers=5, prefetch_factor=2,
            # For model
            latent_dim=100, epochs=100,
            learning_rate=1e-4, scheduler_patience=10,
            # For logging and saving
            model_save_dir: Optional[str] = "model_params/gan",
            log_dir: Optional[str] = "runs/gan"
    ):
        self.dataframe = dataframe
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.learning_rate = learning_rate

        # Determine the shape of the EEG data
        self.eeg_dim = dataframe.iloc[0]["EEG"].shape[0]
        generator_initial_layers.append(self.eeg_dim)

        # Initialize the GAN components with dynamic layers
        self.generator = Generator(self.latent_dim, generator_initial_layers).to("cuda")
        self.discriminator = Discriminator(self.eeg_dim, discriminator_layers).to("cuda")

        # Loss function and optimizers
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

        # Learning rate scheduler
        self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_G, mode='min',
            factor=0.1, patience=scheduler_patience
        )
        self.scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_D, mode='min',
            factor=0.1, patience=scheduler_patience
        )

        # Logging and saving
        self.model_save_dir = model_save_dir
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.model_save_path = os.path.join(self.model_save_dir, "gan_model")

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None

    def get_data_loader(self, shuffle=True):
        return DataLoader(
            EEGGanDataset(self.dataframe),
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def train(self):
        dataloader = self.get_data_loader(shuffle=True)
        overall_start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            batch_count = len(dataloader)

            for i, (real_eeg, _) in enumerate(dataloader):
                real_eeg = real_eeg.float().to("cuda")
                batch_size = real_eeg.size(0)

                # Generate labels
                valid = torch.ones(batch_size, device="cuda")
                fake = torch.zeros(batch_size, device="cuda")

                # Train Generator
                self.optimizer_G.zero_grad()
                z = torch.randn(batch_size, self.latent_dim, 1, device="cuda")
                generated_eeg = self.generator(z)
                g_loss = self.criterion(self.discriminator(generated_eeg), valid)
                g_loss.backward()
                self.optimizer_G.step()

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_loss = self.criterion(self.discriminator(real_eeg), valid)
                fake_loss = self.criterion(self.discriminator(generated_eeg.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

                if i % 50 == 0:
                    print(f"[Epoch {epoch}/{self.epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            # Average losses for the epoch
            avg_g_loss = epoch_g_loss / batch_count
            avg_d_loss = epoch_d_loss / batch_count

            # Scheduler steps
            self.scheduler_G.step(avg_g_loss)
            self.scheduler_D.step(avg_d_loss)

            # Log metrics to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Loss/Generator", avg_g_loss, epoch)
                self.writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
                self.writer.add_scalar("Learning Rate/Generator", self.scheduler_G.optimizer.param_groups[0]['lr'],
                                       epoch)
                self.writer.add_scalar("Learning Rate/Discriminator", self.scheduler_D.optimizer.param_groups[0]['lr'],
                                       epoch)

            # Save model at intervals
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                self._save_model(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {str(timedelta(seconds=epoch_elapsed_time))[:-3]}. "
                  f"Generator loss: {avg_g_loss:.4f}, Discriminator loss: {avg_d_loss:.4f}")

        overall_elapsed_time = time.time() - overall_start_time
        overall_formatted_time = str(timedelta(seconds=overall_elapsed_time))[:-3]
        print(f"Training completed in {overall_formatted_time}.")
        self.writer.close() if self.writer is not None else None

    def augment_data(self):
        dataloader = self.get_data_loader(shuffle=False)

        new_records = []
        for real_eeg, verdict in dataloader:
            batch_size = real_eeg.size(0)
            z = torch.randn(batch_size, self.latent_dim, 1, device="cuda")
            generated_eeg = self.generator(z).cpu().detach().numpy()

            for i in range(batch_size):
                new_records.append({
                    "Subject": "Generated",
                    "Trial": -1,
                    "Trial_Prefix": "GAN",
                    "Trial_Rep": 1,
                    "Date": "N/A",
                    "Start_Frame": 0,
                    "EEG": generated_eeg[i],
                    "Verdict": verdict[i],
                })
        augmented_df = pd.DataFrame(new_records)

        return pd.concat([self.dataframe, augmented_df], ignore_index=True)

    def _save_model(self, epoch):
        model_dicts = {
            "epoch": epoch,
            "model_state_dict": {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict()
            },
            "optimizer_state_dict": {
                "optimizer_G": self.optimizer_G.state_dict(),
                "optimizer_D": self.optimizer_D.state_dict()
            },
        }
        torch.save(model_dicts, f"{self.model_save_path}__epoch_{epoch}.pt")

    def load_model(self, epoch):
        model_path = f"{self.model_save_path}__epoch_{epoch}.pt"
        print(f"Trying to load model from {model_path}")

        model_dicts = torch.load(model_path)
        model_state_dict = model_dicts["model_state_dict"]
        self.generator.load_state_dict(model_state_dict["generator"])
        self.discriminator.load_state_dict(model_state_dict["discriminator"])
        optimizer_state_dict = model_dicts["optimizer_state_dict"]
        self.optimizer_G.load_state_dict(optimizer_state_dict["optimizer_G"])
        self.optimizer_D.load_state_dict(optimizer_state_dict["optimizer_D"])

        print(f"Loaded model from {model_path}")
