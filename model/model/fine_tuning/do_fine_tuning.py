import os
import time
from datetime import timedelta
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from model.common.encoders import TimeFrequencyEncoder, CrossSpaceProjector
from model.common.loss import NTXentLoss
from model.fine_tuning.classifier import EmotionClassifier
from model.fine_tuning.gcn import GCN


class FineTuning:
    SAVED_MODEL_DIR_ADD = "_saved"
    MODEL_ADD = "finetuned_model"

    def __init__(
            self, data_loader, data_loader_eval, sampling_frequency, num_classes,
            # The trained encoders and projectors
            ET: TimeFrequencyEncoder, EF: TimeFrequencyEncoder, PT: CrossSpaceProjector, PF: CrossSpaceProjector,
            *,
            num_channels=62,
            device=None, finetuning_model_save_dir="model_params/finetuning",
            log_dir: Optional[str] = "runs/finetuning",  # Added log_dir
            # Parameters from the paper
            epochs=20, lr=5e-4, weight_decay=3e-4,
            alpha=0.1, beta=0.1, gamma=1.0, temperature=0.5,
            projectors_output_dim=128,
            gcn_hidden_dims=None, gcn_k_order=3, delta=0.2,
            classifier_hidden_dims=None,
            overwrite_training=False, to_train=True,
    ):
        if gcn_hidden_dims is None:
            gcn_hidden_dims = [128, 64]
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [1024, 128]

        self.train_batch_size = len(data_loader) if data_loader is not None else 0
        self.eval_batch_size = len(data_loader_eval)

        self.data_loader = data_loader
        self.data_loader_eval = data_loader_eval
        self.sampling_frequency = sampling_frequency
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.projectors_output_dim = projectors_output_dim
        self.gcn_k_order = gcn_k_order
        self.delta = delta
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = log_dir  # Added log_dir

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
            input_dim=projectors_output_dim * 2,
            hidden_dims=gcn_hidden_dims,
            k_order=self.gcn_k_order
        ).to(self.device)

        self.classifier = EmotionClassifier(
            input_dim=num_channels * gcn_hidden_dims[-1],
            hidden_dims=classifier_hidden_dims,
            output_dim=self.num_classes
        ).to(self.device)

        # Define optimizers with L2 penalty
        self.optimizer = optim.Adam(
            (list(self.ET.parameters()) + list(self.EF.parameters()) +
             list(self.PT.parameters()) + list(self.PF.parameters()) +
             list(self.gcn.parameters()) + list(self.classifier.parameters())),
            lr=self.lr,
            weight_decay=weight_decay
        )

        self.nt_xent_calculator = NTXentLoss(temperature=self.temperature)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        self.model_save_dir = finetuning_model_save_dir
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

    def _do_train_epoch(self, epoch):
        epoch_loss = 0
        with tqdm(self.data_loader, desc=f"Epoch {epoch}", leave=False) as pbar:
            for xT, xT_augmented, xF, xF_augmented, y in pbar:
                xT, xT_augmented, xF, xF_augmented, y = self._move_to_device(
                    xT, xT_augmented, xF, xF_augmented, y
                )

                # Reset the optimizers
                self.optimizer.zero_grad()

                y, logits, L, LT, LF, LA, Lcls = self._forward_pass(
                    xT, xT_augmented, xF, xF_augmented, y
                )
                epoch_loss += L.item()

                # Backpropagation
                L.backward()
                self.optimizer.step()

                # Update tqdm progress bar with the current loss
                pbar.set_description_str(f"Epoch {epoch}, Loss: {L.item():.4f}")

        return epoch_loss / self.train_batch_size

    def do_eval_epoch(self, epoch=None, *, all_accuracies=False):
        if epoch is None:
            base_desc = "Doing evaluation"
        else:
            base_desc = f"Doing evaluation at Epoch {epoch}"

        all_ok_predictions = None
        correct, total = 0, 0
        eval_loss = 0
        with torch.no_grad(), tqdm(self.data_loader_eval, desc=base_desc, leave=False) as pbar:
            self.ET.eval()
            self.EF.eval()
            self.PT.eval()
            self.PF.eval()
            self.gcn.eval()
            self.classifier.eval()

            for xT, xT_augmented, xF, xF_augmented, y in pbar:
                y, logits, L, LT, LF, LA, Lcls = self._forward_pass(
                    xT, xT_augmented, xF, xF_augmented, y, compute_grad=False
                )
                eval_loss += L.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)

                # Update tqdm progress bar with the current loss
                pbar.set_description_str(
                    f"{base_desc} -> "
                    f"Current Loss: {eval_loss / total:.4f}, "
                    f"Current Accuracy: {correct / total:.2%}"
                )

                if all_accuracies:
                    if all_ok_predictions is None:
                        all_ok_predictions = []
                    all_ok_predictions.append(predictions == y)

            self.ET.train()
            self.EF.train()
            self.PT.train()
            self.PF.train()
            self.gcn.train()
            self.classifier.train()

        if all_accuracies:
            all_ok_predictions = torch.cat(all_ok_predictions)

        eval_accuracy = correct / total
        avg_eval_loss = eval_loss / self.eval_batch_size
        return all_ok_predictions, eval_accuracy, avg_eval_loss

    def train(self, *, update_after_every_epoch=True, force_train=False):
        if not self.to_train:
            raise Exception("to_train must be True to train model")
        if self._trained and not force_train:
            raise Exception("Trying to train an already trained model!")

        writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None
        overall_start_time = time.time()

        with tqdm(total=self.epochs, desc="Fine-Tuning Progress", leave=False, unit="epoch") as overall_pbar:
            for epoch in range(1, self.epochs + 1):
                epoch_start_time = time.time()  # TODO: Do timing with a `with` block

                # Do a train loop
                avg_epoch_loss = self._do_train_epoch(epoch)

                # Save the model every 1 epoch
                self._save_model(epoch)

                # Evaluate accuracy
                _, eval_accuracy, avg_eval_loss = self.do_eval_epoch(epoch)

                epoch_duration = time.time() - epoch_start_time

                # Log metrics to TensorBoard
                if writer is not None:
                    writer.add_scalar("Train Loss/epoch", avg_epoch_loss, epoch)
                    writer.add_scalar("Eval Loss/epoch", avg_eval_loss, epoch)
                    writer.add_scalar("Eval Accuracy/epoch", eval_accuracy, epoch)
                    writer.add_scalar("Time/epoch", epoch_duration, epoch)

                # Update overall progress bar
                if update_after_every_epoch:
                    overall_pbar.set_description_str(
                        f"Epoch {epoch},"
                        f"Train Loss: {avg_epoch_loss:.4f},"
                        f"Eval Accuracy: {eval_accuracy:.4f},"
                        f"Eval Loss: {avg_eval_loss:.4f}"
                    )
                overall_pbar.update(1)

        overall_duration = time.time() - overall_start_time
        overall_formatted_time = str(timedelta(seconds=overall_duration))[:-3]
        print(
            f"Fine-tuning completed. "
            f"Time taken: {overall_formatted_time}, "
            f"Final Train Loss: {avg_epoch_loss:.4f}, "
            f"Final Eval Accuracy: {eval_accuracy:.4f}"
            f"Final Eval Loss: {avg_eval_loss:.4f}"
        )

        if writer is not None:
            writer.close()  # Close the TensorBoard writer

        # Rename folder to keep the training "saved"
        os.rename(self.model_save_dir, self.model_final_save_dir)
        self._trained = True

        return

    def _forward_pass(self, xT, xT_augmented, xF, xF_augmented, y, compute_grad=True):
        xT, xT_augmented, xF, xF_augmented, y = self._move_to_device(
            xT, xT_augmented, xF, xF_augmented, y
        )
        channel_count = xT.size(1)

        # Compute separate losses channel by channel, initialized as tensors
        LT = torch.tensor(0.0, device=self.device, requires_grad=compute_grad)
        LF = torch.tensor(0.0, device=self.device, requires_grad=compute_grad)
        LA = torch.tensor(0.0, device=self.device, requires_grad=compute_grad)
        zT_list, zF_list = [], []

        for i in range(channel_count):
            hT, LT_i = self._compute_time_contrastive_loss(
                xT[:, i, :],
                xT_augmented[:, i, :]
            )
            hF, LF_i = self._compute_frequency_contrastive_loss(
                xF[:, i, :],
                xF_augmented[:, i, :]
            )
            zT, zF, LA_i = self._compute_alignment_loss(hT, hF)

            LT = LT + LT_i
            LF = LF + LF_i
            LA = LA + LA_i
            zT_list.append(zT)
            zF_list.append(zF)

        # Average the losses over all channels
        LT = LT / channel_count
        LF = LF / channel_count
        LA = LA / channel_count

        # Stack the projected embeddings
        zT = torch.stack(zT_list, dim=1)
        zF = torch.stack(zF_list, dim=1)

        # Combine and process through GCN and classifier
        Z = torch.cat([zT, zF], dim=-1)
        adj_matrix = self.gcn.build_adjacency_matrix(Z)
        gcn_output = self.gcn(Z, adj_matrix)
        logits = self.classifier(gcn_output.flatten(start_dim=1))

        # Calculate classification loss and overall loss
        Lcls = self.cross_entropy_loss(logits, y)
        L = self.alpha * (LT + LF) + self.beta * LA + self.gamma * Lcls

        return y, logits, L, LT, LF, LA, Lcls

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
                "PF": self.PF.state_dict(),
                "gcn": self.gcn.state_dict(),
                "classifier": self.classifier.state_dict(),
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(model_dicts, f"{self.model_save_path}__epoch_{epoch}.pt")

    def load_model(self, epoch, *, allow_logs=True):
        if self.model_final_save_path is None:
            raise Exception("Tried loading model with no path provided!")

        model_path = f"{self.model_final_save_path}__epoch_{epoch}.pt"
        if allow_logs:
            print(f"Trying to load model from {model_path}")

        model_dicts = torch.load(model_path)
        model_state_dict = model_dicts["model_state_dict"]
        for model_type, state_dict in model_state_dict.items():
            model = getattr(self, model_type)
            if not isinstance(model, torch.nn.modules.module.Module):
                raise ValueError(f"Unknown model type: {type(model)}")
            model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(model_dicts["optimizer_state_dict"])
        if "gcn_state_dict" in model_dicts:
            self.gcn.load_state_dict(model_dicts["gcn_state_dict"])
        if "classifier_state_dict" in model_dicts:
            self.classifier.load_state_dict(model_dicts["classifier_state_dict"])

        self._trained = True
        if allow_logs:
            print(f"Loaded model from {model_path}")
