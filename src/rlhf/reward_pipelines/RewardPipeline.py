import gc

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import sys

sys.path.append("/home/st-gorbatovski/sollama/")

from src.mpnet_reward.models.helper_fucntions import (
    get_embeddings,
    compute_embeddings_sim,
)


class RewardPipeline:
    def __init__(self, model_name, accelerator, length_penalty_config):
        self.model_name = model_name
        self.accelerator = accelerator

        self.reward_model = AutoModel.from_pretrained(
            model_name,
            # device_map=
        )
        self.length_penalty_config = length_penalty_config
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_model.eval()
        self.reward_model = accelerator.prepare(self.reward_model)

    def __call__(self, q_list, a_list, batch_size, reward_type):
        return self.get_rewards(q_list, a_list, batch_size, reward_type)

    @torch.no_grad()
    def get_rewards(self, q_list, a_list, batch_size, reward_type="dot_prod"):
        q_tokenized = self.reward_tokenizer(
            q_list, padding=False, max_length=512, truncation=True
        )
        a_tokenized = self.reward_tokenizer(
            a_list, padding=False, max_length=512, truncation=True
        )

        q_tokenized = self.tokenize_to_list(q_tokenized)
        a_tokenized = self.tokenize_to_list(a_tokenized)

        q_dataloader = DataLoader(q_tokenized, batch_size, collate_fn=self.collate_fn)
        a_dataloader = DataLoader(a_tokenized, batch_size, collate_fn=self.collate_fn)

        q_dataloader = self.accelerator.prepare(q_dataloader)
        a_dataloader = self.accelerator.prepare(a_dataloader)
        rewards = []

        for q_batch, a_batch in zip(q_dataloader, a_dataloader):
            q_embeddings = get_embeddings(self.reward_model, q_batch)
            a_embeddings = get_embeddings(self.reward_model, a_batch)

            # q_embeddings = self.accelerator.gather(q_embeddings.detach()).cpu()
            # a_embeddings = self.accelerator.gather(a_embeddings.detach()).cpu()

            q_embeddings = q_embeddings.detach().cpu()
            a_embeddings = a_embeddings.detach().cpu()

            batch_rewards = compute_embeddings_sim(q_embeddings, a_embeddings)
            batch_rewards = batch_rewards[reward_type]

            if self.length_penalty_config:
                batch_rewards = self.apply_length_penalty(batch_rewards, a_batch)

            del a_embeddings, q_embeddings, a_batch, q_batch
            gc.collect()
            torch.cuda.empty_cache()

            rewards.extend(batch_rewards)

        return list(torch.tensor(rewards).unsqueeze(1))

    def apply_length_penalty(self, rewards, batch):
        lengths = batch["attention_mask"].sum(dim=1).tolist()
        penalties = [
            self.linear_length_penalty(
                length,
                self.length_penalty_config["alpha1"],
                self.length_penalty_config["alpha2"],
                self.length_penalty_config["lower_bound_length"],
                self.length_penalty_config["upper_bound_length"],
            )
            for length in lengths
        ]
        adjusted_rewards = [
            reward - penalty for reward, penalty in zip(rewards, penalties)
        ]
        return adjusted_rewards

    @staticmethod
    def linear_length_penalty(
        length, alpha1=1, alpha2=0.05, lower_bound_length=40, upper_bound_length=128
    ):
        if length < lower_bound_length:
            return alpha1 * (lower_bound_length - length)
        elif length > upper_bound_length:
            return alpha2 * (length - upper_bound_length)
        else:
            return 0

    def collate_fn(self, batch):
        # batch - это список словарей, возвращаемых токенизатором
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        # Делаем паддинг до одинаковой длины
        padded_data = self.reward_tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=8,
            max_length=512,
        )

        input_ids = padded_data["input_ids"]
        attention_mask = padded_data["attention_mask"]

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def tokenize_to_list(self, tokenized):
        return [
            {"input_ids": input_ids, "attention_mask": attention_mask}
            for input_ids, attention_mask in zip(
                tokenized["input_ids"], tokenized["attention_mask"]
            )
        ]
