import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import sys
import json

from E5_base import TextClassifier
from transformers import AutoTokenizer
from datasets import load_dataset

def gen_batch(df, batch_size):
    length = len(df)
    for i in range(0, length, batch_size):
        yield df.iloc[i : i + batch_size]

def inference(df, classifier, tokenizer, threshold, batch_size):
    tags_classification = [
        "Data Science and Machine Learning",
        "Database and SQL",
        "GUI and Desktop Applications",
        "Networking and APIs",
        "Other",
        "Python Basics and Environment",
        "System Administration and DevOps",
        "Web Development",
    ]
    
    results = []

    classifier.eval()  # set the model to evaluation mode

    gen = gen_batch(df, batch_size)

    for batch in tqdm(gen, total=int(len(df) / batch_size)):
        text_batch = (
            batch["Title"] + " " + batch["Question"]
        )  # combine 'Title' and 'Question'
        encoded_batch = tokenizer(
            text_batch.tolist(),
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():  # don't track gradients during the inference
            encoded_batch = {k: v.to(classifier.model.device) for k, v in encoded_batch.items()}
            output = classifier(encoded_batch)

        probs = F.sigmoid(output)  # get probabilities

        batch_results = {tag: (probs[:, idx] > threshold).int().tolist() for idx, tag in enumerate(tags_classification)}
        results.append(pd.DataFrame(batch_results))
    
    results_df = pd.concat(results, ignore_index=True)
    df = pd.concat([df, results_df], axis=1)

    return df


def main():
    parser = argparse.ArgumentParser(description='Perform inference on a given dataset with a trained model.')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for class assignment')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the inference')
    parser.add_argument('--device', type=str, default="cuda:0", help='Devide to run model')
    args = parser.parse_args()

    sys.path.append("/home/st-gorbatovski/sollama/e5_classifier")

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")
    classifier = TextClassifier("intfloat/e5-base", n_inputs=768)

    print("Loading model weights...")
    state_dict = torch.load(args.model_weights)
    state_dict = state_dict["state_dict"]
    new_state_dict = dict()
    del state_dict["model.model.embeddings.position_ids"]

    for key, value in state_dict.items():
        new_state_dict[key[6:]] = value

    classifier.load_state_dict(new_state_dict)
    classifier.to(device)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset)
    df = pd.concat(
        [
            dataset["train"].to_pandas(),
            dataset["validation"].to_pandas(),
            dataset["test"].to_pandas(),
        ],
        ignore_index=True,
    )

    print("Starting inference...")
    classified_df = inference(df, classifier, tokenizer, args.threshold, args.batch_size)
    print("Inference complete. Saving results...")
    classified_df.to_json(args.output, orient='records')
    print(f"Results saved to {args.output}")
    
if __name__ == "__main__":
    main()
