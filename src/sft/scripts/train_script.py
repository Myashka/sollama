import click
import yaml
from yaml import CLoader
import wandb
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
)
from transformers import (
    # GenerationConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

import sys
import os

from src.sft.data import make_train_dataset  # , make_inference_dataset
from src.sft.utils import load_model, set_random_seed, SavePeftModelCallback

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
    print(ddp)
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        config["model"]["device_map"] = device_map

    model, tokenizer = load_model(config["model"])
    if config["model"]["load_in_8bit"]:
        model = prepare_model_for_kbit_training(model)

    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    # prepare LoRA model for training
    if not config["model"].get("peft_model_id"):
        lora_config = LoraConfig(task_type="CAUSAL_LM", **config["lora_config"])
        model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # make datasets
    train_dataset = make_train_dataset(
        tokenizer=tokenizer,
        split=config["data"]["train_split"],
        **config["data"]["dataset"],
    )
    eval_dataset = make_train_dataset(
        tokenizer=tokenizer,
        split=config["data"]["val_split"],
        **config["data"]["dataset"],
    )

    # make generatoin config
    # generation_config = GenerationConfig(**config["generation_config"])

    # training_args = Seq2SeqTrainingArguments(generation_config=generation_config, **config["training_argiments"])
    training_args = TrainingArguments(
        ddp_find_unused_parameters=False if ddp else None,
        **config["training_arguments"],
    )

    callbacks = [SavePeftModelCallback] if config["lora_config"] else []

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        data_collator=DataCollatorForTokenClassification(
            tokenizer,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=8,
        ),
    )
    if config["training_arguments"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.config.use_cache = not config["training_arguments"]["gradient_checkpointing"]

    if config["model"]["load_in_8bit"]:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(config["training_arguments"]["resume_from_checkpoint"])
    tokenizer.save_pretrained(config["training_arguments"]["output_dir"])
    model.save_pretrained(config["training_arguments"]["output_dir"])


if __name__ == "__main__":
    main()
