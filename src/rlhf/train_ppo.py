import click
import yaml
from yaml import CLoader
import gc

import torch
from tqdm import tqdm

tqdm.pandas()

from accelerate import Accelerator
from data import collator, make_dataset
from peft import LoraConfig
from reward_pipelines import RewardPipeline
from transformers import LlamaTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed, create_reference_model
import os

def fix_tokenizer(tokenizer):
    # Fixing broken tokenizers
    special_tokens = dict()
    for token_id in range(1000):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if tokenizer.pad_token_id in (None, tokenizer.vocab_size) and "pad" in token:
            special_tokens["pad_token"] = token
        if tokenizer.bos_token_id in (None, tokenizer.vocab_size) and "<s>" in token:
            special_tokens["bos_token"] = token
        if tokenizer.eos_token_id in (None, tokenizer.vocab_size) and "</s>" in token:
            special_tokens["eos_token"] = token
        if tokenizer.unk_token_id in (None, tokenizer.vocab_size) and "unk" in token:
            special_tokens["unk_token"] = token
        if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "sep" in token:
            special_tokens["sep_token"] = token

    if (
        tokenizer.sep_token_id in (None, tokenizer.vocab_size)
        and "bos_token" in special_tokens
    ):
        special_tokens["sep_token"] = special_tokens["bos_token"]

    if (
        tokenizer.pad_token_id in (None, tokenizer.vocab_size)
        and "pad_token" not in special_tokens
    ):
        if tokenizer.unk_token_id is not None:
            special_tokens["pad_token"] = tokenizer.unk_token
        else:
            special_tokens["pad_token"] = "<|pad|>"

    if (
        tokenizer.sep_token_id in (None, tokenizer.vocab_size)
        and "sep_token" not in special_tokens
    ):
        if tokenizer.bos_token_id is not None:
            special_tokens["sep_token"] = tokenizer.bos_token
        else:
            special_tokens["sep_token"] = "<|sep|>"

    tokenizer.add_special_tokens(special_tokens)

    print("Vocab size: ", tokenizer.vocab_size)
    print("PAD: ", tokenizer.pad_token_id, tokenizer.pad_token)
    print("BOS: ", tokenizer.bos_token_id, tokenizer.bos_token)
    print("EOS: ", tokenizer.eos_token_id, tokenizer.eos_token)
    print("UNK: ", tokenizer.unk_token_id, tokenizer.unk_token)
    print("SEP: ", tokenizer.sep_token_id, tokenizer.sep_token)
    return tokenizer


def fix_model(model, tokenizer, use_resize=False):
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config.pad_token_id is not None

    bos_candidates = (
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id,
    )
    for bos_candidate in bos_candidates:
        model.config.bos_token_id = bos_candidate
        if bos_candidate is not None:
            break
    assert model.config.bos_token_id is not None
    model.config.decoder_start_token_id = model.config.bos_token_id

    eos_candidates = (tokenizer.eos_token_id, tokenizer.sep_token_id)
    for eos_candidate in eos_candidates:
        model.config.eos_token_id = eos_candidate
        if eos_candidate is not None:
            break
    assert model.config.eos_token_id is not None

    if use_resize:
        model.resize_token_embeddings(len(tokenizer))

    print(f"PAD ID: {model.config.pad_token_id}")
    print(f"BOS ID: {model.config.bos_token_id}")
    print(f"EOS ID: {model.config.eos_token_id}")

    return model



@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        args_config = yaml.load(f, Loader=CLoader)

    config = PPOConfig(**args_config["ppo_config"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    data_config = args_config["data"]
    reward_config = args_config["reward"]
    save_config = args_config["save"]

    generation_kwargs = args_config["generation_config"]

    config.data_config = data_config
    config.save_config = save_config
    config.generation_kwargs = config
    config.reward_config = reward_config

    set_seed(config.seed)

    current_device = Accelerator().local_process_index

    lora_config = LoraConfig(
        **args_config["lora_config"],
        task_type="CAUSAL_LM",
    )

    tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    tokenizer = fix_tokenizer(tokenizer)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map={"": current_device},
        peft_config=lora_config,
    )
    model = fix_model(model, tokenizer)
    # ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    dataset = make_dataset(tokenizer=tokenizer, **data_config)

    print("Trainer start")
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )
    print("Trainer done")

    reward_pipe = RewardPipeline(
        reward_config["reward_model_name"], ppo_trainer.accelerator, reward_config["length_penalty_config"]
    )

    best_reward = -100
    model = torch.compile(model)

    print(ppo_trainer.current_device)
    print(config.total_ppo_epochs)
    for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        # print(f"current_devic: {torch.cuda.current_device()}")        
        # print(f"Epoch {epoch} size:", end='')
        # print(len(batch["input_ids"]))
        if epoch >= config.total_ppo_epochs:
            break

        query_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )

        rewards = reward_pipe(
            batch["query"],
            batch["response"],
            reward_config["batch_size"],
            reward_config["reward_type"],
        )

        #### Run PPO step
        # print(f"query_tensors: {len(query_tensors)}")
        # print(f"response_tensors: {len(response_tensors)}")
        # print(f"rewards: {len(rewards)}")
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        mean_reward = torch.mean(torch.tensor(rewards)).item()

        del batch
        del rewards
        torch.cuda.empty_cache()
        gc.collect()

        if (epoch + 1) % save_config["save_interval"] == 0:
            ppo_trainer.save_pretrained(
                save_config["checkpoint_dir"]
                + f"_step_{epoch}-mean_reward_{round(mean_reward, 2)}"
            )

        if mean_reward > best_reward:
            ppo_trainer.save_pretrained(
                save_config["best_checkpoint_dir"]
                + f"_step_{epoch}-mean_reward_{round(mean_reward, 2)}"
            )

            best_reward = mean_reward

    ppo_trainer.save_pretrained(
        save_config["checkpoint_dir"]
        + f"_step_{epoch}-mean_reward_{round(mean_reward, 2)}"
    )
    print("Train finished")


if __name__ == "__main__":
    main()
