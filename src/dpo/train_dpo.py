import click
import yaml
from yaml import CLoader

import torch
from tqdm import tqdm
tqdm.pandas()

from transformers import TrainingArguments
from peft import LoraConfig
from trl import DPOTrainer

from src.dpo.data import make_datasets
from src.sft.utils import load_model, set_random_seed

import os
os.environ["WANDB_PROJECT"] = "SO_LLAMA"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        args_config = yaml.load(f, Loader=CLoader)
    set_random_seed(args_config["training_arguments"]["seed"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    data_config = args_config["data"]
    lora_config = args_config.get("lora_config", None)
    model_config = args_config["model"]
    training_arguments = args_config["training_arguments"]


    ddp = world_size != 1
    print(ddp)
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model_config["device_map"] = device_map

    print("Start load the model")
    model, tokenizer = load_model(model_config)
    print("Model loaded")
    model.config.use_cache = False

    if args_config["ignore_bias_buffers"]:

        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    if lora_config:
        peft_config = LoraConfig(task_type="CAUSAL_LM", **lora_config)

    train_dataset, eval_dataset = make_datasets(
        tokenizer=tokenizer, do_train=True, **data_config
    )

    training_args = TrainingArguments(
        ddp_find_unused_parameters=False if ddp else None,
        **training_arguments,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=args_config["beta"],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=512,
        max_length=768,
    )
    # dpo_trainer.model.gradient_checkpointing_enable()
    dpo_trainer.model.enable_input_require_grads()
    dpo_trainer.ref_model.enable_input_require_grads()
    dpo_trainer.model = torch.compile(dpo_trainer.model)
    dpo_trainer.train()
    dpo_trainer.model.save_pretrained(args_config["training_arguments"]["output_dir"])
    dpo_trainer.tokenizer.save_pretrained(args_config["training_arguments"]["output_dir"])

if __name__ == "__main__":
    main()