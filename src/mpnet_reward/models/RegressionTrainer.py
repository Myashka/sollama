from transformers import Trainer
from sklearn.metrics import precision_score, recall_score
import torch.nn as nn
    
class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_p"],  attention_mask=inputs["attention_mask_p"])[0]
        rewards_k = model(input_ids=inputs["input_ids_n"], attention_mask=inputs["attention_mask_n"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


def compute_reg_acc_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = predictions.reshape((1, -1))[0]
    labels = labels.reshape((1, -1))[0]

    preds = (predictions >= 0).astype(int)
    target = (labels >= 0).astype(int)

    accuracy = sum(preds == target) / len(target)
    precision = precision_score(target, preds)
    recall = recall_score(target, preds)

    rewards = predictions.tolist()

    mae = abs(predictions - labels).mean()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "rewards": rewards,
        "mae": mae,
    }
