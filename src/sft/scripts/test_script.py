import click
import wandb
import yaml
import torch

# from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from yaml import CLoader

import sys

sys.path.append("/home/st-gorbatovski/sollama/")

from src.sft.utils import load_model, set_random_seed
from src.sft.data import make_inference_dataset
from src.sft.models import eval_model


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    model, tokenizer = load_model(config["eval"]["model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    run = wandb.init(
        **config["wandb_config"],
        config=config["eval"],
    )

    set_random_seed(config["eval"]["seed"])

    # accelerator = Accelerator()

    columns_to_save = config["eval"]["data"]['columns_to_save']

    test_dataset = make_inference_dataset(tokenizer=tokenizer, **config["eval"]["data"])
    dataset_qa = test_dataset.remove_columns(["input_ids", "attention_mask"])
    
    test_dataset = test_dataset.remove_columns(columns_to_save)

    dataloader_ids = DataLoader(
        test_dataset,
        batch_size=config["eval"]["batch_size"],
        collate_fn=DataCollatorForTokenClassification(
            tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
    )

    dataloader_qa = DataLoader(dataset_qa, batch_size=config["eval"]["batch_size"])

    # model, dataloader = accelerator.prepare(model, dataloader)

    eval_model(
        run,
        model,
        dataloader_ids,
        dataloader_qa,
        tokenizer,
        config["eval"]["generate_config"],
        config["log_config"],
        config["eval"]["compute_metrics"],
        columns_to_save
    )

    run.finish()


if __name__ == "__main__":
    main()
