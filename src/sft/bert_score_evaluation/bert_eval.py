import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import gc
from torchmetrics.functional.text.bert import bert_score
import os


def get_device(device):
    return device if torch.cuda.is_available() else "cpu"


def get_scores(device, target, preds, batch_size):
    assert len(target) == len(preds)
    preds = [pred if pred == pred else "" for pred in preds]
    print(f"Running evaluation on {device}")
    f1_scores = []
    for i in tqdm(range(0, len(preds), batch_size)):
        t_batch = target[i : i + batch_size]
        p_batch = preds[i : i + batch_size]
        f1_scores.extend(
            bert_score(p_batch, t_batch, num_threads=2, device=device)["f1"]
        )
        del t_batch
        del p_batch
        gc.collect()
        torch.cuda.empty_cache()
    return f1_scores


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = get_device(args.device)
    print(device)
    print('Loading data...')
    df = pd.read_csv(args.file_path)
    target = df[args.target_column].tolist()
    preds = df[args.pred_column].tolist()
    print(f"Target length: {len(target)}")
    print(f"Prediction length: {len(preds)}")
    print("Start computing...")
    f1_scores = get_scores(device, target, preds, args.batch_size)
    print(f"Count of BERT_F1 scores: {len(f1_scores)}")
    df[args.bert_scores_column] = f1_scores
    df.to_csv(args.file_path, index=False)
    print(f"Result saved to {args.file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", default="path", type=str, help="Path to the file to be processed"
    )
    parser.add_argument(
        "--device", default="cuda:0", type=str, help="Device to run BERT"
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size for processing"
    )
    parser.add_argument(
        "--target_column", default="Answer", type=str, help="Name of the target column"
    )
    parser.add_argument(
        "--pred_column",
        default="Generated Answer",
        type=str,
        help="Name of the predictions column",
    )
    parser.add_argument(
        "--bert_scores_column",
        default="BERT Score",
        type=str,
        help="Name of the column to save BERT scores",
    )
    args = parser.parse_args()
    main(args)
