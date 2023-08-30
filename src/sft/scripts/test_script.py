import click
import wandb
import yaml
import torch

# from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import set_seed
from yaml import CLoader

import sys

sys.path.append("/home/st-gorbatovski/sollama/")

from src.sft.utils import load_model
from src.sft.data import make_inference_dataset
from src.sft.models import eval_model


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    run = wandb.init(
        **config["wandb_config"],
        config=config["eval"],
    )

    model, tokenizer = load_model(config["eval"]["model"])

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    set_seed(config["eval"]["seed"])

    # accelerator = Accelerator()

    test_dataset = make_inference_dataset(tokenizer=tokenizer, **config["eval"]["data"])
    dataloader = DataLoader(test_dataset, batch_size=1)

    # model, dataloader = accelerator.prepare(model, dataloader)

    model = torch.compile(model)

    eval_model(
        run,
        model,
        dataloader,
        tokenizer,
        config["eval"]["generate_config"],
        config["log_config"],
        config["eval"]["compute_metrics"],
    )

    run.finish()


if __name__ == "__main__":
    main()
