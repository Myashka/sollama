import torch

import torch.nn.functional as F

def l2_distance(x1, x2):
    """Calculate Euclidean (L2) distance between two tensors."""
    return F.pairwise_distance(x1, x2, p=2)

def cos_distance(x1, x2):
    """Calculate Cosine distance between two tensors."""
    return 1 - F.cosine_similarity(x1, x2, dim=1)

def cos_similarity_loss(x1, x2, eps=1e-6):
    cos_similarity = F.cosine_similarity(x1, x2, dim=1)
    loss = -torch.log((1 - cos_similarity) / 2 + eps) / 6
    return loss.mean()

def cos_similarity(x1, x2):
    """Calculate Cosine similarity between two tensors."""
    return F.cosine_similarity(x1, x2, dim=1)

def dot_prod_sim(x1, x2):
    """Calculate dot product similarity between two tensors."""
    return (x1 * x2).sum(dim=1)

def compute_embedding_metrics(predictions):
    """Compute different metrics for embeddings based on the class."""
    embeddings_a = torch.tensor(predictions[0][0])
    embeddings_p = torch.tensor(predictions[0][1])
    embeddings_n = torch.tensor(predictions[0][2])
    is_gen_k = torch.tensor(predictions[0][3])
    is_par_j = torch.tensor(predictions[0][4])
    is_par_k = torch.tensor(predictions[0][5])

    mask_gen_k = is_gen_k == 1
    mask_par_k = is_par_k == 1
    mask_par_j = is_par_j == 1
    mask_so_k = (~mask_gen_k) & (~mask_par_k)
    mask_so_j = (~mask_par_j) 

    masks_and_suffixes = [
        (mask_so_j & mask_gen_k, "so_gen"),
        (mask_so_j & mask_par_k, "so_par"),
        (mask_so_j & mask_so_k, "so_so"),
        (mask_par_j & mask_par_k, "par_par"),
        (mask_par_j & mask_so_k, "par_so"),
        (mask_par_j & mask_gen_k, "par_gen")
    ]
    
    dot_prod_p = dot_prod_sim(embeddings_a, embeddings_p)
    dot_prod_n = dot_prod_sim(embeddings_a, embeddings_n)
    dot_prod_p_n = dot_prod_sim(embeddings_p, embeddings_n)

    # Вычисление евклидова расстояния
    euclidean_distance_p = l2_distance(embeddings_a, embeddings_p)
    euclidean_distance_n = l2_distance(embeddings_a, embeddings_n)

    metrics = {
        'mean_dot_prod_gap': (dot_prod_p - dot_prod_n).mean().item(),
        'mean_euclidean_distance': (euclidean_distance_p - euclidean_distance_n).mean().item(),
        'pos_rewards': dot_prod_p.tolist(),
        'neg_rewards': dot_prod_n.tolist(),
        "accuracy": (dot_prod_p > dot_prod_n).float().mean().item()
    }

    accuracies = []

    for mask, suffix in masks_and_suffixes:
        if mask.sum() > 0:
            accuracy_c = (dot_prod_p[mask] > dot_prod_n[mask]).float().mean().item()
            accuracies.append(accuracy_c)
            metrics[f'accuracy_{suffix}'] = accuracy_c
            metrics[f'dot_prod_p_{suffix}'] = dot_prod_p[mask].mean().item()
            metrics[f'dot_prod_n_{suffix}'] = dot_prod_n[mask].mean().item()
            metrics[f'dot_prod_p_n_{suffix}'] = dot_prod_p_n[mask].mean().item()

    metrics['avg_class_accuracy'] = sum(accuracies)/len(accuracies)

    return metrics