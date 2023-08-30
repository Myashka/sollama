import gc
import os
import pandas as pd
import torch
import wandb
from torchmetrics import SacreBLEUScore
from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm
from transformers import GenerationConfig

from src.sft.utils import log_metrics_histograms, log_table, save_csv


def generate_outputs(model, batch_ids, generation_config):
    with torch.autocast("cuda"):
        output_tokens = (
            model.generate(
                input_ids=batch_ids["input_ids"],
                attention_mask=batch_ids["attention_mask"],
                generation_config=generation_config,
            )
            .cpu()
            .numpy()
        )
    return output_tokens


def decode_outputs(tokenizer, output_tokens, input_ids, generation_config):
    input_ids = [
        ids for ids in input_ids for _ in range(generation_config.num_return_sequences)
    ]
    # outputs = [str(gen) for gen in tokenizer.batch_decode(output_tokens, skip_special_tokens=True)]
    outputs = []
    for sample_output_tokens, sample_input_ids in zip(output_tokens, input_ids):
        sample_output_tokens = sample_output_tokens[len(sample_input_ids) :]
        gen_answer = tokenizer.decode(sample_output_tokens, skip_special_tokens=True)
        gen_answer = gen_answer.replace("</s>", "").strip()
        outputs.append(gen_answer)
    return outputs


def compute_scores(result, rouge, bleu):
    rouge_scores = [
        rouge(gen_answ, answ)
        for gen_answ, answ in zip(result["Generated Answer"], result["Answer"])
    ]
    bleu_scores = [
        bleu(gen_answ, answ)
        for gen_answ, answ in zip(result["Generated Answer"], result["Answer"])
    ]

    metrics = {
        "ROUGE_1": [score["rouge1_fmeasure"].item() for score in rouge_scores],
        "ROUGE_2": [score["rouge2_fmeasure"].item() for score in rouge_scores],
        "ROUGE_L": [score["rougeL_fmeasure"].item() for score in rouge_scores],
        "BLEU": [score.item() for score in bleu_scores],
    }
    return metrics


def log_results(results, log_config, run):
    os.makedirs(log_config["dir"], exist_ok=True)

    save_csv(results, f"{log_config['dir']}/{log_config['file_name']}")
    log_table(run, log_config["file_name"], results)


def eval_model(
    run,
    model,
    dataloader_ids,
    dataloader_qa,
    tokenizer,
    generate_config,
    log_config,
    compute_metrics=True,
    columns_to_save=["Question", "Answer"],
):
    """
    Evaluates the model by generating responses for the input data and calculates evaluation metrics.

    :param run: wandb run object to log the metrics and results.
    :param model: The language model to be evaluated.
    :param dataloader_ids: Dataloader object that provides batches of ids data for evaluation.
    :param dataloader_qa: Dataloader object that provides batches of textual data for evaluation.
    :param tokenizer: Tokenizer object to decode the generated responses.
    :param generate_config: Configuration dict for the model's generate function.
    :param log_config: Configuration dict specifying the frequency of logging and the location to save the logs.
    :param compute_metrics: Boolean indicating whether to compute evaluation metrics (default: True).

    :return: None. This function does not return any value. The results are logged using wandb and saved as CSV files.
    """

    rouge = ROUGEScore()
    bleu = SacreBLEUScore(1, lowercase=True)

    results = pd.DataFrame()
    generation_config = GenerationConfig(**generate_config)

    for i, (batch_ids, batch_qa) in enumerate(tqdm(zip(dataloader_ids, dataloader_qa), total=len(dataloader_qa))):
        batch_ids = {k: v.to(model.device) for k, v in batch_ids.items()}
        output_tokens = generate_outputs(model, batch_ids, generation_config)
        outputs = decode_outputs(
            tokenizer, output_tokens, batch_ids["input_ids"], generation_config
        )

        id_sequence = range(
            i * len(outputs) // generation_config.num_return_sequences,
            (i + 1) * len(outputs) // generation_config.num_return_sequences,
        )

        ids = [number for number in id_sequence for _ in range(generation_config.num_return_sequences)]

        result_dict = dict()
        for column in columns_to_save:
            if column in ("Score", "Users Score", "Q_Id", "log_score"):
                result_dict[column] = [value.item() for value in batch_qa[column] for _ in range(generation_config.num_return_sequences)]
            else:
                result_dict[column] = [value for value in batch_qa[column] for _ in range(generation_config.num_return_sequences)]

        result_dict["Gen_Q_Id"] = ids
        result_dict["Generated Answer"] = outputs

        result = pd.DataFrame(result_dict)

        if compute_metrics:
            metrics = compute_scores(result, rouge, bleu)

            for key, value in metrics.items():
                result[key] = value

        results = pd.concat([results, result], ignore_index=True)

        del output_tokens, result
        torch.cuda.empty_cache()
        gc.collect()

        if (i + 1) % log_config["save_steps"] == 0:
            log_results(results, log_config, run)

            # Clear the results for the next iteration
            results = pd.DataFrame()
            gc.collect()

    if not results.empty:
        log_results(results, log_config, run)

    if compute_metrics:
        metrics_df = pd.read_csv(f"{log_config['dir']}/{log_config['file_name']}")
        log_metrics_histograms(run, log_config["file_name"], metrics_df)

    artifact = wandb.Artifact(
        log_config["file_name"].replace(".csv", ""), type="dataset"
    )
    artifact.add_file(f"{log_config['dir']}/{log_config['file_name']}")
    run.log_artifact(artifact)
