import click
import wandb
import yaml
import torch

# from accelerate import Accelerator
from torch.utils.data import DataLoader
from yaml import CLoader

import sys

from src.mpnet_reward.utils import load_model, set_random_seed
from src.mpnet_reward.data import make_datasets, RewardDataCollatorWithPadding
from src.mpnet_reward.models.eval_mpnet_model import eval_mpnet_model


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    model, tokenizer = load_model(config["eval"]["model"])
    model.eval()

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    run = wandb.init(
        **config["wandb_config"],
        config=config["eval"],
    )

    set_random_seed(config["eval"]["seed"])

    # accelerator = Accelerator()
    columns_to_save = config["eval"]["data"]["columns_to_save"]

    test_dataset = make_datasets(
        tokenizer=tokenizer, do_train=False, **config["eval"]["data"]
    )
    ids_cols = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_p",
        "attention_mask_p",
        "input_ids_n",
        "attention_mask_n",
        "is_par_j",
        "is_par_k",
        "is_gen_k",
    ]
    dataset_qa = test_dataset.remove_columns(ids_cols[:-3])
    test_dataset = test_dataset.remove_columns(
        [col for col in test_dataset.column_names if col not in ids_cols]
    )

    dataloader_ids = DataLoader(
        test_dataset,
        batch_size=config["eval"]["batch_size"],
        collate_fn=RewardDataCollatorWithPadding(
            tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
    )

    dataloader_qa = DataLoader(dataset_qa, batch_size=config["eval"]["batch_size"])

    # model, dataloader = accelerator.prepare(model, dataloader)

    eval_mpnet_model(
        run,
        model,
        dataloader_ids,
        dataloader_qa,
        config["log_config"],
        columns_to_save,
    )

    run.finish()


if __name__ == "__main__":
    main()
