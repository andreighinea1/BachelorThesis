import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from model.common.encoders import TimeFrequencyEncoder, CrossSpaceProjector
from model.common.loss import NTXentLoss
from model.fine_tuning.classifier import EmotionClassifier
from model.fine_tuning.gcn import GCN


class FineTuning:
    def __init__(
            self, data_loader, sampling_frequency, num_classes,
            # The trained encoders and projectors
            ET: TimeFrequencyEncoder, EF: TimeFrequencyEncoder, PT: CrossSpaceProjector, PF: CrossSpaceProjector,
            *,
            device=None, finetuning_model_save_dir="model_params/finetuning",
            # Parameters from the paper
            epochs=20, lr=5e-4, l2_norm_penalty=3e-4,
            alpha=0.1, beta=0.1, gamma=1.0, temperature=0.05,
            encoders_output_dim=200, projectors_output_dim=128,
            gcn_hidden_dims=None, gcn_k_order=3, delta=0.2,
            classifier_hidden_dims=None,
            overwrite_training=False,
    ):
        if gcn_hidden_dims is None:
            gcn_hidden_dims = [128, 64]
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [1024, 128]

        self.data_loader = data_loader
        self.sampling_frequency = sampling_frequency
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.l2_norm_penalty = l2_norm_penalty
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.encoders_output_dim = encoders_output_dim
        self.projectors_output_dim = projectors_output_dim
        self.gcn_k_order = gcn_k_order
        self.delta = delta
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models initialization
        self.ET = ET.to(self.device)
        self.ET.train()
        self.EF = EF.to(self.device)
        self.EF.train()
        self.PT = PT.to(self.device)
        self.PT.train()
        self.PF = PF.to(self.device)
        self.PF.train()

        self.gcn = GCN(
            input_dim=self.projectors_output_dim * 2,
            hidden_dims=gcn_hidden_dims,
            k_order=self.gcn_k_order
        ).to(self.device)

        self.classifier = EmotionClassifier(
            input_dim=gcn_hidden_dims[-1],
            hidden_dims=classifier_hidden_dims,
            output_dim=self.num_classes
        ).to(self.device)

        # Define optimizers with L2 penalty
        self.optimizer = optim.Adam(
            (list(self.ET.parameters()) + list(self.EF.parameters()) +
             list(self.PT.parameters()) + list(self.PF.parameters()) +
             list(self.gcn.parameters()) + list(self.classifier.parameters())),
            lr=self.lr,
            weight_decay=self.l2_norm_penalty
        )

        self.nt_xent_calculator = NTXentLoss(temperature=self.temperature)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        if finetuning_model_save_dir is not None:
            if not os.path.exists(finetuning_model_save_dir):
                os.makedirs(finetuning_model_save_dir)
            if not overwrite_training and os.listdir(finetuning_model_save_dir):
                raise Exception(f"Model folder not empty, probably already trained: {finetuning_model_save_dir}")
            self.model_save_path = os.path.join(finetuning_model_save_dir, f"finetuned_model")
        else:
            self.model_save_path = None

    def train(self):
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            with tqdm(self.data_loader, desc=f"Epoch {epoch}", leave=False) as pbar:
                for xT, xT_augmented, xF, xF_augmented, y in pbar:
                    xT, xT_augmented, xF, xF_augmented, y = self._move_to_device(xT, xT_augmented, xF, xF_augmented, y)
                    channel_count = xT.size(1)

                    # Reset the optimizers
                    self.optimizer.zero_grad()

                    # Compute separate losses channel by channel
                    LT, LF, LA, zT_list, zF_list = 0., 0., 0., [], []
                    for i in range(channel_count):
                        hT, LT_i = self._compute_time_contrastive_loss(
                            xT[:, i, :].unsqueeze(1),
                            xT_augmented[:, i, :].unsqueeze(1)
                        )
                        hF, LF_i = self._compute_frequency_contrastive_loss(
                            xF[:, i, :].unsqueeze(1),
                            xF_augmented[:, i, :].unsqueeze(1)
                        )
                        zT, zF, LA_i = self._compute_alignment_loss(hT, hF)

                        LT += LT_i
                        LF += LF_i
                        LA += LA_i
                        zT_list.append(zT)
                        zF_list.append(zF)

                    # Average the losses over all channels
                    LT /= channel_count
                    LF /= channel_count
                    LA /= channel_count

                    # Stack the projected embeddings
                    zT = torch.cat(zT_list, dim=1)
                    zF = torch.cat(zF_list, dim=1)

                    # Combine and process through GCN and classifier
                    Z = torch.cat([zT, zF], dim=-1)
                    adj_matrix = self.gcn.build_adjacency_matrix(Z)
                    gcn_output = self.gcn(Z, adj_matrix)
                    logits = self.classifier(gcn_output.flatten())  # TODO: With flatten or not?

                    # Calculate classification loss
                    Lcls = self.cross_entropy_loss(logits, y)
                    L = self.alpha * (LT + LF) + self.beta * LA + self.gamma * Lcls
                    epoch_loss += L.item()

                    # Backpropagation
                    L.backward()
                    self.optimizer.step()

                    # Update tqdm progress bar with the current loss
                    pbar.set_description_str(f"Epoch {epoch}, Loss: {L.item():.4f}")

            # Save the model every 10 epochs and at the last epoch
            if epoch % 10 == 0 or epoch == self.epochs:
                self._save_model(epoch)

            print(f"Epoch {epoch}, Average Loss: {epoch_loss / len(self.data_loader):.4f}")

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
        model_dicts = {
            "epoch": epoch,
            "model_state_dict": {
                "ET": self.ET.state_dict(),
                "EF": self.EF.state_dict(),
                "PT": self.PT.state_dict(),
                "PF": self.PF.state_dict()
            },
            "gcn_state_dict": self.gcn.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(model_dicts, f"{self.model_save_path}__epoch_{epoch}.pt")
