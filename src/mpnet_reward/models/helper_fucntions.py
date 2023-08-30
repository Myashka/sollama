import torch
import torch.nn.functional as F

from .metrics import l2_distance, cos_similarity, dot_prod_sim


def mean_pooling(model_output, attention_mask):
    token_embeddings = (
        model_output.last_hidden_state
    )  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_embeddings(model, batch_ids, postfix=""):
    with torch.no_grad():
        output = model(
            input_ids=batch_ids[f"input_ids{postfix}"],
            attention_mask=batch_ids[f"attention_mask{postfix}"],
            return_dict=True,
        )
        embeddings = mean_pooling(output, batch_ids[f"attention_mask{postfix}"])
    return embeddings


def get_embeddings_for_batch(model, batch_ids):
    embeddings_q = get_embeddings(model, batch_ids, postfix="_a")
    embeddings_j = get_embeddings(model, batch_ids, postfix="_p")
    embeddings_k = get_embeddings(model, batch_ids, postfix="_n")

    return embeddings_q, embeddings_j, embeddings_k


def compute_embeddings_sim(embedding_a, embedding_b, postfix=""):
    sim_dict = dict()
    sim_dict[f"dot_prod{postfix}"] = dot_prod_sim(embedding_a, embedding_b).tolist()
    sim_dict[f"cos_sim{postfix}"] = cos_similarity(embedding_a, embedding_b).tolist()

    embedding_a = F.normalize(embedding_a, p=2, dim=1)
    embedding_b = F.normalize(embedding_b, p=2, dim=1)
    sim_dict[f"euclidean{postfix}"] = l2_distance(embedding_a, embedding_b).tolist()
    return sim_dict
