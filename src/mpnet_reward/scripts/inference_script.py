import torch
import pandas as pd

import argparse
from tqdm import tqdm
import gc

from src.mpnet_reward.utils import load_model, set_random_seed
from src.mpnet_reward.models.helper_fucntions import (
    get_embeddings,
    compute_embeddings_sim,
)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def merge_dicts(dict1, dict2):
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            result[key].extend(value)
        else:
            result[key] = value
    return result


def get_rewards(
    model, tokenizer, q_list, a_list, batch_size, max_length_q, max_length_a, postfix
):
    print(f"Running evaluation on {model.device}")
    rewards_dict = dict()
    for i in tqdm(range(0, len(q_list), batch_size)):
        batch_q = q_list[i : i + batch_size]
        batch_a = a_list[i : i + batch_size]

        batch_q = tokenizer(
            batch_q, padding="longest", max_length=max_length_q, truncation=True, return_tensors='pt',
        )
        batch_a = tokenizer(
            batch_a, padding="longest", max_length=max_length_a, truncation=True, return_tensors='pt',
        )

        batch_q = {k: v.to(model.device) for k, v in batch_q.items()}
        q_embeddings = get_embeddings(model, batch_q)

        batch_a = {k: v.to(model.device) for k, v in batch_a.items()}
        a_embeddings = get_embeddings(model, batch_a)

        batch_rewards_dict = compute_embeddings_sim(q_embeddings, a_embeddings, postfix)
        rewards_dict = merge_dicts(rewards_dict, batch_rewards_dict)

        del batch_q, batch_a, a_embeddings, q_embeddings
        torch.cuda.empty_cache()
        gc.collect()

    return rewards_dict


def main(args):
    config = {
        "use_title": args.use_title,
        "t_col": args.t_col,
        "q_col": args.q_col,
        "a_col": args.a_col,
        "max_length_q": args.max_length_q,
        "max_length_a": args.max_length_a,
        "postfix": args.postfix,
        "batch_size": args.batch_size,
        "file_path": args.file_path,
        "model": {
            "name": args.model_name,
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "padding_side": args.padding_side,
        },
        "seed": args.seed,
    }

    print(f"Configuration: {config}")


    model, tokenizer = load_model(config["model"])
    model.eval()
    model.config.use_cache = True
    
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    set_random_seed(config["seed"])

    df = pd.read_csv(config["file_path"], lineterminator="\n")
    print(f"Length of data: {len(df)}")

    if config["use_title"]:
        df["t_q_col"] = df[config["t_col"]] + "\n" + df[config["q_col"]]
        q_list = df["t_q_col"].tolist()
    else:
        q_list = df[config["q_col"]].tolist()
    a_list = df[config["a_col"]].tolist()

    rewards_dict = get_rewards(
        model, tokenizer, q_list, a_list,
        config["batch_size"],
        config["max_length_q"],
        config["max_length_a"],
        config["postfix"],
    )

    for key, value in rewards_dict.items():
        df[key] = value
    
    if config["use_title"]:
        df = df.drop(columns=['t_q_col'])

    df.to_csv(config["file_path"], index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for evaluating rewards")
    parser.add_argument("--use_title", type=bool, default=False, help="Use title in the questions")
    parser.add_argument("--t_col", type=str, help="Title column")
    parser.add_argument("--q_col", type=str, help="Questions column")
    parser.add_argument("--a_col", type=str, help="Answers column")
    parser.add_argument("--max_length_q", type=int, help="Maximum length for questions")
    parser.add_argument("--max_length_a", type=int, help="Maximum length for answers")
    parser.add_argument("--postfix", type=str, help="Postfix to the end of additional columns")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--file_path", type=str, help="File path")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--torch_dtype", type=str, help="Torch data type")
    parser.add_argument("--device_map", type=str, help="Device map")
    parser.add_argument("--padding_side", type=str, default="right", help="Padding side")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    main(args)
