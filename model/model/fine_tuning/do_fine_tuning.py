import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from model.common.loss import NTXentLoss
from model.fine_tuning.classifier import EmotionClassifier
from model.fine_tuning.gcn import GCN


class FineTuning:
    def __init__(
            self, data_loader, sampling_frequency, num_classes,
            # The trained encoders and projectors
            ET, EF, PT, PF,
            *,
            device=None, finetuning_model_save_dir="model_params/finetuning",
            # Parameters from the paper
            epochs=20, lr=5e-4, l2_norm_penalty=3e-4,
            alpha=0.1, beta=0.1, gamma=1.0, temperature=0.05,
            encoders_output_dim=200, projectors_output_dim=128,
            num_layers=2, nhead=8, gcn_hidden_dim1=128,
            gcn_hidden_dim2=64, k_order=3, delta=0.2,
    ):
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
        self.num_layers = num_layers
        self.nhead = nhead
        self.gcn_hidden_dim1 = gcn_hidden_dim1
        self.gcn_hidden_dim2 = gcn_hidden_dim2
        self.k_order = k_order
        self.delta = delta
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models initialization
        self.ET = ET.to(self.device)
        self.EF = EF.to(self.device)
        self.PT = PT.to(self.device)
        self.PF = PF.to(self.device)

        self.gcn = GCN(
            input_dim=self.projectors_output_dim * 2,
            hidden_dim1=self.gcn_hidden_dim1,
            hidden_dim2=self.gcn_hidden_dim2,
            k_order=self.k_order
        ).to(self.device)

        self.classifier = EmotionClassifier(
            input_dim=self.gcn_hidden_dim2,
            hidden_dims=[1024, 128],
            output_dim=self.num_classes
        ).to(self.device)

        # Define optimizers with L2 penalty
        self.optimizer = optim.Adam(
            list(self.gcn.parameters()) + list(self.classifier.parameters()),
            lr=self.lr,
            weight_decay=self.l2_norm_penalty
        )

        self.nt_xent_calculator = NTXentLoss(temperature=self.temperature)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        if not os.path.exists(finetuning_model_save_dir):
            os.makedirs(finetuning_model_save_dir)
        self.model_save_path = os.path.join(finetuning_model_save_dir, f"finetuned_model")

    def _load_pretrained_models(self, ET_path, EF_path, PT_path, PF_path):
        self.ET.load_state_dict(torch.load(ET_path))
        self.EF.load_state_dict(torch.load(EF_path))
        self.PT.load_state_dict(torch.load(PT_path))
        self.PF.load_state_dict(torch.load(PF_path))
        self.ET.eval()
        self.EF.eval()
        self.PT.eval()
        self.PF.eval()

    def train(self):
        for epoch in range(1, self.epochs + 1):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch}", leave=False)
            epoch_loss = 0
            for batch in pbar:
                x, y = batch['data'].to(self.device), batch['label'].to(self.device)
                self.optimizer.zero_grad()

                zT, zF, LT, LF, LA = self._generate_embeddings_and_losses(x)

                Z = torch.cat([zT, zF], dim=-1)
                adj_matrix = self._construct_adjacency_matrix(Z, self.delta)

                gcn_output = self.gcn(Z, adj_matrix)
                logits = self.classifier(gcn_output)

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

    def _generate_embeddings_and_losses(self, x):
        zT_list = []
        zF_list = []
        LT_list = []
        LF_list = []
        LA_list = []

        for channel in range(x.size(1)):
            hT = self.ET(x[:, channel, :])
            hT_augmented = self.ET(x[:, channel, :])
            LT = self.nt_xent_calculator.calculate_loss(hT, hT_augmented)

            hF = self.EF(x[:, channel, :])
            hF_augmented = self.EF(x[:, channel, :])
            LF = self.nt_xent_calculator.calculate_loss(hF, hF_augmented)

            zT = self.PT(hT)
            zF = self.PF(hF)
            LA = self.nt_xent_calculator.calculate_loss(zT, zF)

            zT_list.append(zT)
            zF_list.append(zF)
            LT_list.append(LT)
            LF_list.append(LF)
            LA_list.append(LA)

        zT = torch.stack(zT_list, dim=1).mean(dim=1)
        zF = torch.stack(zF_list, dim=1).mean(dim=1)
        LT = torch.stack(LT_list).mean()
        LF = torch.stack(LF_list).mean()
        LA = torch.stack(LA_list).mean()

        return zT, zF, LT, LF, LA

    @staticmethod
    def _construct_adjacency_matrix(Z, delta=0.2):
        """
        Construct the adjacency matrix based on cosine similarity.

        Parameters:
        - Z (torch.Tensor): The node features matrix (N x F), where N is the number of nodes (channels)
          and F is the feature dimension.
        - delta (float): The threshold value to determine the adjacency matrix entries.

        Returns:
        - adj_matrix (torch.Tensor): The constructed adjacency matrix (N x N).
        """
        # Calculate the cosine similarity between each pair of node features
        similarity_matrix = F.cosine_similarity(Z.unsqueeze(1), Z.unsqueeze(0), dim=-1)

        # Apply the adjacency matrix formula
        adj_matrix = torch.exp(similarity_matrix - 1)
        adj_matrix[similarity_matrix < delta] = delta

        return adj_matrix

    def _save_model(self, epoch):
        model_dicts = {
            "gcn_state_dict": self.gcn.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
        }
        torch.save(model_dicts, f"{self.model_save_path}__epoch_{epoch}.pt")
        print(f"Saved model at epoch {epoch}")
