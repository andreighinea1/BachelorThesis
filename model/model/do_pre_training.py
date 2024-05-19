import os

import torch
import torch.optim as optim
from tqdm.auto import tqdm

from model.encoders import TimeFrequencyEncoder, CrossSpaceProjector
from model.loss import NTXentLoss


# Parameters from the paper
class PreTraining:
    def __init__(
            self, data_loader, sampling_frequency, *,
            device=None, pretraining_model_save_dir="model_params/pretraining",

            epochs=1000,
    ):
        self.data_loader = data_loader
        self.sampling_frequency = sampling_frequency
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.epochs = epochs
        self.lr = 3e-4
        self.l2_norm_penalty = 3e-4
        self.alpha = 0.2
        self.beta = 1.0
        self.temperature = 0.05

        self.encoders_output_dim = 200
        self.projectors_output_dim = 128
        self.num_layers = 2
        self.nhead = 8

        # Models initialization
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

        # Define optimizers with L2 penalty
        self.optimizer = optim.Adam(
            (list(self.ET.parameters()) + list(self.EF.parameters()) +
             list(self.PT.parameters()) + list(self.PF.parameters())),
            lr=self.lr,
            weight_decay=self.l2_norm_penalty  # L2-norm penalty coefficient
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=10
        )

        self.nt_xent_calculator = NTXentLoss(temperature=self.temperature)

        if not os.path.exists(pretraining_model_save_dir):
            os.makedirs(pretraining_model_save_dir)
        self.model_save_path = os.path.join(pretraining_model_save_dir, f"pretrained_model")

    def train(self):
        for epoch in range(1, self.epochs + 1):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch}", leave=False)
            epoch_loss = 0
            for xT, xT_augmented, xF, xF_augmented in pbar:
                xT, xT_augmented, xF, xF_augmented = self._move_to_device(xT, xT_augmented, xF, xF_augmented)

                # Reset the optimizers
                self.optimizer.zero_grad()

                # Compute separate losses
                hT, LT = self._compute_time_contrastive_loss(xT, xT_augmented)
                hF, LF = self._compute_frequency_contrastive_loss(xF, xF_augmented)
                LA = self._compute_alignment_loss(hT, hF)

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

            print(f"Epoch {epoch},"
                  f"Average Loss: {epoch_loss / len(self.data_loader):.4f},"
                  f"Learning Rate: {self.scheduler.get_last_lr():.6f}")

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
        return LA

    def _save_model(self, epoch):
        model_dicts = {
            "ET_state_dict": self.ET.state_dict(),
            "EF_state_dict": self.EF.state_dict(),
            "PT_state_dict": self.PT.state_dict(),
            "PF_state_dict": self.PF.state_dict(),
        }
        torch.save(model_dicts, f"{self.model_save_path}__epoch_{epoch}.pt")
        print(f"Saved model at epoch {epoch}")
