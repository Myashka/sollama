import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm
import gc
import random
import argparse
import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_csv(df, file_path):
    """
    Saves the DataFrame as a CSV file. If the file already exists, the DataFrame is appended without repeating the header.

    :param df: pandas DataFrame to be saved.
    :param file_path: String representing the path to the file where the DataFrame should be saved.

    :return: None. This function does not return any value. The DataFrame is saved as a CSV file at the specified location.
    """
    if os.path.exists(file_path):
        mode = "a"
        header = False
    else:
        mode = "w"
        header = True
    df.to_csv(file_path, mode=mode, header=header, index=False)


def paraphrase(
    texts,
    model,
    tokenizer,
    num_beams=2,
    num_beam_groups=2,
    num_return_sequences=3,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=512,
):
    texts = [f"paraphrase: {text}" for text in texts]

    input_ids = tokenizer(
        texts,
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids

    outputs = model.generate(
        input_ids.to(model.device),
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        max_length=256,
        diversity_penalty=diversity_penalty,
    ).cpu().numpy()

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

def main(args):
    set_random_seed(123)
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_path)
    df = pd.concat(
        [dataset["train"].to_pandas(), dataset["validation"].to_pandas(), dataset["test"].to_pandas()],
        ignore_index=True,
    )

    # Инициализация модели и токенизатора
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map={"": args.device})

    for n, p in model.named_parameters():
        p.requires_grad = False

    results = pd.DataFrame()

    print("Starting paraphrasing...")
    for i in tqdm(range(0, len(df), args.batch_size), desc="Processing batches"):
        batch = df.iloc[i:i + args.batch_size]
        paraphrased_batch = paraphrase(batch['Answer'].tolist(),
                                       model,
                                       tokenizer,
                                       args.num_beams,
                                       args.num_beam_groups,
                                       args.num_return_sequences, 
                                       args.repetition_penalty,
                                       args.diversity_penalty,
                                       args.no_repeat_ngram_size,
                                       args.temperature,
                                       args.max_length)
        
        id_sequence = range(
                i * len(paraphrased_batch) // args.num_return_sequences,
                (i + 1) * len(paraphrased_batch) // args.num_return_sequences,
            )
        ids = [number for number in id_sequence for _ in range(args.num_return_sequences)]
        result_dict = {column: [value for value in batch[column] for _ in range(args.num_return_sequences)] for column in df.columns}
        result_dict["Par_A_Id"] = ids
        result_dict["Paraphrased Answer"] = paraphrased_batch
        result = pd.DataFrame(result_dict)
        results = pd.concat([results, result], ignore_index=True)

        if (i + 1) % args.save_frequency == 0:
            print(f"Saving {len(results)} rows to {args.save_file_path}...")
            save_csv(results, args.save_file_path)
            print(f"Saved successfully.")
            results = pd.DataFrame()
            gc.collect()
            torch.cuda.empty_cache()

    save_csv(results, args.save_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paraphrase generation script.")
    
    parser.add_argument("--dataset_path", default="Myashka/SO-Python_basics_QA-filtered-2023-tanh_score", help="Path to the dataset.")
    parser.add_argument("--model_name", default="humarin/chatgpt_paraphraser_on_T5_base", help="Name or path of the paraphrasing model.")
    parser.add_argument("--device", default="cuda:0", help="Device for model inference.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--save_frequency", type=int, default=5, help="Frequency of saving the results.")
    parser.add_argument("--save_file_path", default="/home/st-gorbatovski/sollama/t5_paraphraser/SO-Python_basics_QA-filtered-2023-tanh_score_paraphrased_data.csv", help="Path to save the results.")
    
    # Add all the parameters for the paraphrase function
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_beam_groups", type=int, default=5)
    parser.add_argument("--num_return_sequences", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=10.0)
    parser.add_argument("--diversity_penalty", type=float, default=3.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    
    main(args)