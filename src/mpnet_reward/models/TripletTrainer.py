from transformers import Trainer
import torch
import torch.nn.functional as F
from .metrics import l2_distance, cos_distance, dot_prod_sim, cos_similarity_loss
from .helper_fucntions import mean_pooling


class TripletTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        anchor_output = model(
            input_ids=inputs["input_ids_a"], attention_mask=inputs["attention_mask_a"]
        )
        positive_output = model(
            input_ids=inputs["input_ids_p"], attention_mask=inputs["attention_mask_p"]
        )
        negative_output = model(
            input_ids=inputs["input_ids_n"], attention_mask=inputs["attention_mask_n"]
        )

        embeddings_a = mean_pooling(anchor_output, inputs["attention_mask_a"])
        embeddings_p = mean_pooling(positive_output, inputs["attention_mask_p"])
        embeddings_n = mean_pooling(negative_output, inputs["attention_mask_n"])

        if self.model.config.normalize:
            embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
            embeddings_p = F.normalize(embeddings_p, p=2, dim=1)
            embeddings_n = F.normalize(embeddings_n, p=2, dim=1)

        so_margin = self.model.config.so_margin
        gen_margin = self.model.config.gen_margin
        similarity_type = self.model.config.similarity_type

        dist_function = self.choose_dis_fucntion(similarity_type)

        is_gen_k = inputs["is_gen_k"]
        unique_classes = torch.unique(is_gen_k)
        loss = 0
        for c in unique_classes:
            mask = is_gen_k == c
            if c == 0:
                margin = so_margin
                loss += (cos_similarity_loss(embeddings_p[mask], embeddings_n[mask]) * self.model.config.a_n_loss_weight)
            else:
                margin = gen_margin
            loss += self.all_batch_loss(
                embeddings_a[mask],
                embeddings_p[mask],
                embeddings_n[mask],
                margin,
                dist_function,
                similarity_type,
            )

        if return_outputs:
            return loss, {
                "embedding_a": embeddings_a,
                "embedding_p": embeddings_p,
                "embedding_n": embeddings_n,
                "is_gen_k": inputs["is_gen_k"],
                "is_par_j": inputs["is_par_j"],
                "is_par_k": inputs["is_par_k"],
            }
        return loss

    def all_batch_loss(
        self,
        embeddings_a,
        embeddings_p,
        embeddings_n,
        margin,
        distance_func,
        similarity_type,
    ):
        if similarity_type in ["euclidean", "cos_dist"]:
            positive_distance = distance_func(embeddings_a, embeddings_p)
            negative_distance = distance_func(embeddings_a, embeddings_n)
            loss = F.relu(positive_distance - negative_distance + margin)
        else:  # dot_prod_sim
            positive_similarity = distance_func(embeddings_a, embeddings_p)
            negative_similarity = distance_func(embeddings_a, embeddings_n)
            loss = F.relu(negative_similarity - positive_similarity + margin)
        return loss.mean()

    def choose_dis_fucntion(self, similarity_type):
        if similarity_type == "euclidean":
            distance_func = l2_distance
        elif similarity_type == "cos_dist":
            distance_func = cos_distance
        else:
            distance_func = dot_prod_sim
        return distance_func
