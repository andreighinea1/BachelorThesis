import os
import time
from datetime import timedelta
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataset_processing.gan.discriminator import Discriminator
from dataset_processing.gan.generator import Generator


class EegGan:
    MODEL_ADD = "gan_model"

    def __init__(
            self, eeg_data: torch.Tensor, generator_layers: List[int], discriminator_layers: List[int], *,
            device=None,
            # For DataLoader
            batch_size=10, num_workers=5, prefetch_factor=2,
            # For model
            lstm_dim=128, use_full_lstm=False, latent_dim=100, epochs=100,
            learning_rate=2e-4, beta1=0.5, beta2=0.999, scheduler_patience=10,
            # For logging and saving
            model_save_dir: Optional[str] = "model_params/gan",
            log_dir: Optional[str] = "runs/gan",
            overwrite_training=False, to_train=True,
    ):
        self.eeg_data = eeg_data
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.learning_rate = learning_rate

        # region Init return values
        self.overall_elapsed_time = None
        self.overall_formatted_time = None
        # endregion

        # region Init basic variables
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = log_dir

        self.model_save_dir = model_save_dir
        self.to_train = to_train
        self._trained = False

        if self.model_save_dir is not None:
            if not os.path.exists(self.model_save_dir):
                os.makedirs(self.model_save_dir)
            if not overwrite_training and to_train and os.listdir(self.model_save_dir):
                raise Exception(f"Model folder not empty, probably already trained: {self.model_save_dir}")

            self.model_save_path = os.path.join(self.model_save_dir, self.MODEL_ADD)

            self.model_final_save_dir = self.model_save_dir + "_saved"
            self.model_final_save_path = os.path.join(self.model_final_save_dir, self.MODEL_ADD)
        else:
            self.model_save_path = None
            self.model_final_save_dir = None
            self.model_final_save_path = None
        # endregion

        # Determine the shape of the EEG data
        num_samples = eeg_data.shape[2]

        # Initialize the GAN components with dynamic layers
        self.generator = Generator(
            latent_dim=self.latent_dim,
            conv_dims=generator_layers,
            lstm_dim=self.lstm_dim,
            eeg_length=num_samples,
            use_full_lstm=use_full_lstm,
        ).to(self.device)
        self.discriminator = Discriminator(
            conv_dims=discriminator_layers,
            lstm_dim=self.lstm_dim,
            eeg_length=num_samples,
            use_full_lstm=use_full_lstm,
        ).to(self.device)

        # Loss function and optimizers
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

        # Learning rate scheduler
        self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_G, mode="min",
            factor=0.1, patience=scheduler_patience
        )
        self.scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_D, mode="min",
            factor=0.1, patience=scheduler_patience
        )

        # MaxAbsScaler for data normalization
        eeg_data_np = eeg_data.numpy().reshape(-1, num_samples)
        self.scaler = MaxAbsScaler()
        self.scaler.fit(eeg_data_np)
        self.eeg_data = torch.tensor(self.scaler.transform(eeg_data_np)).view(eeg_data.shape)

    def _get_data_loader(self, shuffle=True):
        return DataLoader(
            TensorDataset(self.eeg_data),
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def train(self, update_after_every_epoch=True, force_train=False):
        if not self.to_train:
            raise Exception("to_train must be True to train model")
        if self._trained and not force_train:
            raise Exception("Trying to train an already trained model!")

        dataloader = self._get_data_loader(shuffle=True)
        writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None
        last_save = 0
        overall_start_time = time.time()

        with tqdm(total=self.epochs, desc="Training Progress", leave=False, unit="epoch") as overall_pbar:
            for epoch in range(1, self.epochs + 1):
                start_time = time.time()
                epoch_g_loss = 0.0
                epoch_d_loss = 0.0
                batch_count = len(dataloader)

                for i, (real_eeg,) in enumerate(dataloader):
                    # noinspection PyUnresolvedReferences
                    real_eeg = real_eeg.float().to(self.device)
                    batch_size = real_eeg.size(0)

                    # Generate labels
                    valid = torch.ones(batch_size, 1, device=self.device)
                    fake = torch.zeros(batch_size, 1, device=self.device)

                    # Train Generator
                    self.optimizer_G.zero_grad()
                    z = torch.randn(batch_size, 1, self.latent_dim, device=self.device)
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

                # Average losses for the epoch
                avg_g_loss = epoch_g_loss / batch_count
                avg_d_loss = epoch_d_loss / batch_count

                # Scheduler steps
                self.scheduler_G.step(avg_g_loss)
                self.scheduler_D.step(avg_d_loss)

                # Log metrics to TensorBoard
                current_lr_G = self.scheduler_G.get_last_lr()[0]
                current_lr_D = self.scheduler_D.get_last_lr()[0]

                # Save model at intervals
                if epoch % 10 == 0 or epoch == self.epochs:
                    self._save_model(epoch)
                    last_save = epoch

                # Calculate the elapsed time for the epoch
                elapsed_time = time.time() - start_time

                if writer is not None:
                    writer.add_scalar("Loss/Generator", avg_g_loss, epoch)
                    writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
                    writer.add_scalar("Learning Rate/Generator", current_lr_G, epoch)
                    writer.add_scalar("Learning Rate/Discriminator", current_lr_D, epoch)
                    writer.add_scalar("Time/epoch", elapsed_time, epoch)

                if update_after_every_epoch:
                    overall_pbar.set_description_str(
                        f"last save: {last_save}, "
                        f"D loss: {avg_d_loss:.4f}, G loss: {avg_g_loss:.4f}, "
                        f"lr_D: {current_lr_D:.0e}, lr_G: {current_lr_G:.0e}"
                    )
                overall_pbar.update(1)

            # Training completed
            self.overall_elapsed_time = time.time() - overall_start_time
            self.overall_formatted_time = str(timedelta(seconds=self.overall_elapsed_time))[:-3]
            print(f"Training completed in {self.overall_formatted_time}.")

        if writer is not None:
            writer.close()

        # Rename folder to keep the training "saved"
        os.rename(self.model_save_dir, self.model_final_save_dir)
        self._trained = True

        return

    def augment_data(self, eeg_data):
        batch_size = eeg_data.size(0)
        z = torch.randn(batch_size, 1, self.latent_dim, device=self.device)
        generated_eeg = self.generator(z).cpu().detach().numpy()
        generated_eeg = self.scaler.inverse_transform(generated_eeg.reshape(batch_size, -1))
        return torch.tensor(generated_eeg).view(batch_size, 1, -1)

    def _save_model(self, epoch):
        if self.model_save_path is None:
            return  # Will not save model

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
        if self.model_final_save_path is None:
            raise Exception("Tried loading model with no path provided!")

        model_path = f"{self.model_save_path}__epoch_{epoch}.pt"
        print(f"Trying to load model from {model_path}")

        model_dicts = torch.load(model_path)
        model_state_dict = model_dicts["model_state_dict"]
        self.generator.load_state_dict(model_state_dict["generator"])
        self.discriminator.load_state_dict(model_state_dict["discriminator"])
        optimizer_state_dict = model_dicts["optimizer_state_dict"]
        self.optimizer_G.load_state_dict(optimizer_state_dict["optimizer_G"])
        self.optimizer_D.load_state_dict(optimizer_state_dict["optimizer_D"])

        self._trained = True
        print(f"Loaded model from {model_path}")
