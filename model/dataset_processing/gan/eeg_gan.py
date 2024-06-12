from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
            latent_dim=100, epochs=100, learning_rate=1e-4,
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

        for epoch in range(self.epochs):
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

                if i % 50 == 0:
                    print(f"[Epoch {epoch}/{self.epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

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
