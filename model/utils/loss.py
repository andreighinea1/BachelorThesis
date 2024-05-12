import torch
import torch.nn.functional as F


def nt_xent_loss(a, b, *, double_negative_similarity=False, temperature=0.05):
    """
    Calculates the NT-Xent loss for a batch of embeddings and their augmented versions, where the positive pair
    consists of each embedding and its augmentation, and negative pairs are computed between the embedding and all
    other non-augmented embeddings in the batch.

    Parameters:
    - z (torch.Tensor): Embeddings from the original EEG signals.
    - z_augmented (torch.Tensor): Corresponding embeddings from the augmented EEG signals.
    - temperature (float): Temperature scaling factor for the softmax.

    Returns:
    - torch.Tensor: The average NT-Xent loss for the batch.
    """
    device = a.device
    batch_size = a.size(0)

    # Normalize the embeddings to use cosine similarity
    a = F.normalize(a, p=2, dim=1).to(device)
    b = F.normalize(b, p=2, dim=1).to(device)

    if double_negative_similarity:
        negative_similarity = torch.cat([a, b], dim=0)
        negative_similarity = F.normalize(negative_similarity, p=2, dim=1).to(device)
    else:
        negative_similarity = a.t()

    # Calculate the cosine similarity between each original and its augmented version (positive pairs)
    # Already normalized, so no need to divide `a * b` by anything
    positive_sim = torch.sum(a * b, dim=1) / temperature

    # Calculate cosine similarity between each original and all other originals (for negatives)
    negative_sim_matrix = torch.mm(a, negative_similarity) / temperature
    # Mask out self-similarities (diagonal elements)
    mask = torch.eye(batch_size, device=device)
    negative_sim_matrix = negative_sim_matrix.masked_fill(mask == 1, float('-inf'))

    # Use log-sum-exp trick to calculate the denominator of the softmax function
    # Ref: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    # max_negative_sim = torch.max(negative_sim_matrix, dim=1, keepdim=True)[0]
    # exp_negative_sim = torch.exp(negative_sim_matrix - max_negative_sim)
    # sum_exp_negative_sim = torch.sum(exp_negative_sim, dim=1, keepdim=True)
    # logsumexp_negatives = torch.log(sum_exp_negative_sim + 1e-6) + max_negative_sim.squeeze()
    logsumexp_negatives = torch.logsumexp(negative_sim_matrix, dim=1)

    # Calculate log probabilities for the positives in relation to the negative similarities
    log_prob = positive_sim - logsumexp_negatives

    # Mean loss across all samples
    loss = -torch.mean(log_prob)

    return loss
