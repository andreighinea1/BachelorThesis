import os
import time
from datetime import timedelta
from typing import Optional

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from model.common.encoders import TimeFrequencyEncoder, CrossSpaceProjector
from model.common.loss import NTXentLoss


class EarlyStopping:
    def __init__(self, patience=100, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, loss, epoch):
        """ If return True => We stop training """

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class PreTraining:
    SAVED_MODEL_DIR_ADD = "_saved"
    MODEL_ADD = "pretrained_model"

    def __init__(
            self, data_loader, sampling_frequency, *,
            device=None,
            pretraining_model_save_dir: Optional[str] = "model_params/pretraining",
            log_dir: Optional[str] = "runs/pretraining",
            scheduler_patience=50, early_stopping_patience=100,
            # Parameters from the paper
            epochs=1000, lr=3e-4, l2_norm_penalty=3e-4,
            alpha=0.2, beta=1.0, temperature=0.05,
            encoders_output_dim=200, projectors_output_dim=128,
            num_layers=2, nhead=8,
            overwrite_training=False, to_train=True,
    ):
        # region Init return values
        self.overall_elapsed_time = None
        self.overall_formatted_time = None
        self.last_epoch_loss = None
        # endregion

        # region Init basic variables
        self.data_loader = data_loader
        self.sampling_frequency = sampling_frequency
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = log_dir

        self.model_save_dir = pretraining_model_save_dir
        self.to_train = to_train
        self._trained = False

        if self.model_save_dir is not None:
            if not os.path.exists(self.model_save_dir):
                os.makedirs(self.model_save_dir)
            if not overwrite_training and to_train and os.listdir(self.model_save_dir):
                raise Exception(f"Model folder not empty, probably already trained: {self.model_save_dir}")

            self.model_save_path = os.path.join(self.model_save_dir, self.MODEL_ADD)

            self.model_final_save_dir = self.model_save_dir + self.SAVED_MODEL_DIR_ADD
            self.model_final_save_path = os.path.join(self.model_final_save_dir, self.MODEL_ADD)
        else:
            self.model_save_path = None
            self.model_final_save_dir = None
            self.model_final_save_path = None
        # endregion

        # region Hyperparameters
        self.epochs = epochs
        self.lr = lr
        self.l2_norm_penalty = l2_norm_penalty
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        self.encoders_output_dim = encoders_output_dim
        self.projectors_output_dim = projectors_output_dim
        self.num_layers = num_layers
        self.nhead = nhead

        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        # endregion

        # region Models initialization
        self.ET = TimeFrequencyEncoder(
            input_dim=self.sampling_frequency,
            output_dim=self.encoders_output_dim,
            num_layers=self.num_layers,
            nhead=self.nhead,
        ).to(self.device)

        self.EF = TimeFrequencyEncoder(
            input_dim=self.sampling_frequency,
            output_dim=self.encoders_output_dim,
            num_layers=self.num_layers,
            nhead=self.nhead,
        ).to(self.device)

        self.PT = CrossSpaceProjector(
            input_dim=self.encoders_output_dim,
            output_dim=self.projectors_output_dim,
        ).to(self.device)

        self.PF = CrossSpaceProjector(
            input_dim=self.encoders_output_dim,
            output_dim=self.projectors_output_dim,
        ).to(self.device)
        # endregion

        # region Define optimizers with L2 penalty + Scheduler + Loss calculator
        self.optimizer = optim.Adam(
            (list(self.ET.parameters()) + list(self.EF.parameters()) +
             list(self.PT.parameters()) + list(self.PF.parameters())),
            lr=self.lr,
            weight_decay=self.l2_norm_penalty  # L2-norm penalty coefficient
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=scheduler_patience
        )

        self.nt_xent_calculator = NTXentLoss(temperature=self.temperature)
        # endregion

    def train(self, *, update_after_every_epoch=True, force_train=False):
        if not self.to_train:
            raise Exception("to_train must be True to train model")
        if self._trained and not force_train:
            raise Exception("Trying to train an already trained model!")

        writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None
        batch_count = len(self.data_loader)
        last_epoch_loss = 0.
        last_save = 0
        overall_start_time = time.time()

        with tqdm(total=self.epochs, desc="Training Progress", leave=False, unit="epoch") as overall_pbar:
            for epoch in range(1, self.epochs + 1):
                start_time = time.time()

                # Calculate on a mini-batch
                epoch_loss = 0
                with tqdm(self.data_loader, desc=f"Epoch {epoch}", leave=False) as pbar:
                    for xT, xT_augmented, xF, xF_augmented, _y in pbar:
                        xT, xT_augmented, xF, xF_augmented = self._move_to_device(
                            xT, xT_augmented, xF, xF_augmented
                        )

                        # Reset the optimizers
                        self.optimizer.zero_grad()

                        # Compute separate losses
                        hT, LT = self._compute_time_contrastive_loss(xT, xT_augmented)
                        hF, LF = self._compute_frequency_contrastive_loss(xF, xF_augmented)
                        _, _, LA = self._compute_alignment_loss(hT, hF)

                        # Compute total loss
                        L = self.alpha * (LT + LF) + self.beta * LA
                        epoch_loss += L.item()

                        # Backpropagation
                        L.backward()
                        self.optimizer.step()

                        # Update tqdm progress bar with the current loss
                        pbar.set_description_str(f"Epoch {epoch}, Loss: {L.item():.4f}")

                # Step the scheduler based on the epoch loss
                self.scheduler.step(epoch_loss)

                # Save the model every 10 epochs and at the last epoch
                if epoch % 10 == 0 or epoch == self.epochs:
                    self._save_model(epoch)
                    last_save = epoch

                # Calculate the elapsed time for the epoch
                elapsed_time = time.time() - start_time

                # Log metrics to TensorBoard
                last_epoch_loss = epoch_loss / batch_count  # Avg loss for the epoch
                current_lr = self.scheduler.get_last_lr()[0]
                if writer is not None:
                    writer.add_scalar("Loss/epoch", last_epoch_loss, epoch)
                    writer.add_scalar("Learning Rate/epoch", current_lr, epoch)
                    writer.add_scalar("Time/epoch", elapsed_time, epoch)

                if update_after_every_epoch:
                    overall_pbar.set_description_str(
                        f"last save: {last_save}, "
                        f"loss: {last_epoch_loss:.4f}, "
                        f"lr: {current_lr:.0e}"
                    )
                overall_pbar.update(1)

                # Early stopping check
                if self.early_stopping(last_epoch_loss, epoch):
                    print(
                        f"Early stopping at epoch {epoch} due to no improvement in loss for "
                        f"{self.early_stopping.patience} epochs."
                    )
                    break

        self.overall_elapsed_time = time.time() - overall_start_time
        self.overall_formatted_time = str(timedelta(seconds=self.overall_elapsed_time))[:-3]
        print(
            f"Pre-training completed. "
            f"Last epoch loss: {last_epoch_loss:.4f}. "
            f"Time taken to train: {self.overall_formatted_time}"
        )
        self.last_epoch_loss = last_epoch_loss

        if writer is not None:
            writer.close()  # Close the TensorBoard writer

        # Rename folder to keep the training "saved"
        os.rename(self.model_save_dir, self.model_final_save_dir)
        self._trained = True

        return

    def _move_to_device(self, *args):
        """ Move batches of data to the `device` """
        return [arg.to(self.device) for arg in args]

    def _compute_time_contrastive_loss(self, xT, xT_augmented):
        """ Time Domain Contrastive Learning """
        hT = self.ET(xT)  # Encode time data
        hT_augmented = self.ET(xT_augmented)  # Encode augmented time data
        LT = self.nt_xent_calculator.calculate_loss(  # Calculate the time-based contrastive loss LT in Eq. 1
            hT,
            hT_augmented
        )
        return hT, LT

    def _compute_frequency_contrastive_loss(self, xF, xF_augmented):
        """ Frequency Domain Contrastive Learning """
        hF = self.EF(xF)  # Encode frequency data
        hF_augmented = self.EF(xF_augmented)  # Encode augmented frequency data
        LF = self.nt_xent_calculator.calculate_loss(  # Calculate the frequency-based contrastive loss LF in Eq. 2
            hF,
            hF_augmented
        )
        return hF, LF

    def _compute_alignment_loss(self, hT, hF):
        zT = self.PT(hT)  # Project into shared latent space
        zF = self.PF(hF)  # Project into shared latent space
        LA = self.nt_xent_calculator.calculate_loss(  # Calculate the alignment loss LA in Eq. 3
            zT,
            zF
        )
        return zT, zF, LA

    def _save_model(self, epoch):
        if self.model_save_path is None:
            return  # Will not save model

        model_dicts = {
            "epoch": epoch,
            "model_state_dict": {
                "ET": self.ET.state_dict(),
                "EF": self.EF.state_dict(),
                "PT": self.PT.state_dict(),
                "PF": self.PF.state_dict()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(model_dicts, f"{self.model_save_path}__epoch_{epoch}.pt")

    def load_model(self, epoch):
        if self.model_final_save_path is None:
            raise Exception("Tried loading model with no path provided!")

        model_path = f"{self.model_final_save_path}__epoch_{epoch}.pt"
        print(f"Trying to load model from {model_path}")

        model_dicts = torch.load(model_path)
        model_state_dict = model_dicts["model_state_dict"]
        for model_type, state_dict in model_state_dict.items():
            model = getattr(self, model_type)
            if not isinstance(model, torch.nn.modules.module.Module):
                raise ValueError(f"Unknown model type: {type(model)}")
            model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(model_dicts["optimizer_state_dict"])

        self._trained = True
        print(f"Loaded model from {model_path}")
