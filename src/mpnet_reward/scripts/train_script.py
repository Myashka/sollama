import click
import yaml
from yaml import CLoader
import wandb
import torch
from transformers import TrainingArguments

import sys
import os

from src.mpnet_reward.data import make_datasets, RewardDataCollatorWithPadding
from src.mpnet_reward.models.TripletTrainer import TripletTrainer
from src.mpnet_reward.utils import (
    load_model,
    set_random_seed,
    SaveModelOutputsCallback,
    freeze_model,
)
from src.mpnet_reward.models.metrics import compute_embedding_metrics

os.environ["WANDB_PROJECT"] = "SO_LLAMA"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    set_random_seed(config["training_arguments"]["seed"])

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        config["model"]["device_map"] = device_map

    model, tokenizer = load_model(config["model"])

    freeze_model(model, config["freeze"])

    model.enable_input_require_grads()

    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    train_dataset, eval_dataset = make_datasets(
        tokenizer=tokenizer, **config["data"]
    )

    model.config.normalize = config["triplet_arguments"]["normalize"]
    model.config.so_margin = config["triplet_arguments"]["so_margin"]
    model.config.gen_margin = config["triplet_arguments"]["gen_margin"]
    model.config.similarity_type = config["triplet_arguments"]["similarity_type"]
    model.config.a_n_loss_weight = config["triplet_arguments"]["a_n_loss_weight"]

    training_args = TrainingArguments(
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if ddp else None,
        **config["training_arguments"],
    )

    callbacks = [SaveModelOutputsCallback()]

    print('Trainer creation')
    trainer = TripletTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=8,
        ),
        compute_metrics=compute_embedding_metrics,
    )
    
    model.config.use_cache = True

    print('Compile model')
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print('Start train')
    trainer.train(config["training_arguments"]["resume_from_checkpoint"])
    tokenizer.save_pretrained(config["training_arguments"]["output_dir"])
    model.save_pretrained(config["training_arguments"]["output_dir"])


if __name__ == "__main__":
    main()
