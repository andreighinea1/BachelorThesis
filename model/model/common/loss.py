import torch
import torch.nn.functional as F


class NTXentLoss:
    def __init__(self, temperature=0.05):
        """
        Initializes the NT-Xent loss calculator.

        Parameters:
            temperature (float):
                Temperature scaling factor for the softmax computation, controlling the sharpness of the distribution.
        """
        self.temperature = temperature

    def calculate_loss(self, a, b):
        """
        Calculates the NT-Xent loss for a batch of embeddings `a` and `b`, where the positive pair
        consists of each (a_i, b_i), with negative pairs (a_i, a_j) and (a_i, b_j).

        Parameters:
            a (torch.Tensor):
                Embeddings from the original EEG signals.
            b (torch.Tensor):
                Corresponding embeddings from the augmented EEG signals,
                or from the other domain in time-frequency contrastive learning.

        Returns:
            torch.Tensor: The average NT-Xent loss for the batch.
        """

        if a.size() != b.size():
            raise ValueError(f"Embeddings must have the same number of embeddings:\n"
                             f"{a.size()} != {b.size()}")

        # Normalize the embeddings to use cosine similarity
        a = F.normalize(a, p=2, dim=-1).to(a.device)
        b = F.normalize(b, p=2, dim=-1).to(a.device)

        # Calculate the cosine similarity between each original and its augmented version (positive pairs)
        # Already normalized, so no need to divide `a * b` by anything
        positive_sim = torch.sum(a * b, dim=-1) / self.temperature

        # Calculate cosine similarity between each original and all other originals (negative pairs)
        logsumexp_negatives_a = self.calculate_logsumexp_negatives(a, a)
        logsumexp_negatives_b = self.calculate_logsumexp_negatives(a, b)

        # Calculate log probabilities for the positives in relation to the negative similarities
        log_prob = positive_sim - logsumexp_negatives_a - logsumexp_negatives_b

        # Mean loss across all samples
        loss = -torch.mean(log_prob)

        return loss

    def calculate_logsumexp_negatives(self, a, negatives):
        """
        Calculates the log-sum-exp of negative cosine similarities for a given batch of embeddings against a set of
        negative samples. This function is used as a helper to compute the denominator of the softmax function in
        NT-Xent loss.

        Parameters:
            a (torch.Tensor):
                Embeddings for which negative similarities need to be calculated.
            negatives (torch.Tensor):
                A tensor of negative samples against which the cosine similarities are to be computed.
                This could be the same as `a` or a different set depending on the context.

        Returns:
            torch.Tensor:
                A tensor of log-sum-exp values representing the negative similarities for each sample in `a`.
        """
        # Calculate cosine similarity between each original and all other originals (for negatives)
        negative_sim_matrix = torch.bmm(a, negatives.transpose(-1, -2)) / self.temperature

        # Mask out self-similarities (diagonal elements)
        mask = torch.eye(negative_sim_matrix.size(-1), device=a.device, dtype=torch.bool)
        negative_sim_matrix = negative_sim_matrix.masked_fill(mask, float('-inf'))

        # Use log-sum-exp trick to calculate the denominator of the softmax function
        logsumexp_negatives = torch.logsumexp(negative_sim_matrix, dim=-1)

        return logsumexp_negatives
