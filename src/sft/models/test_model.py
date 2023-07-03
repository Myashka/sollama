import gc

import pandas as pd
import torch
import wandb
from torchmetrics import ROUGEScore, SacreBLEUScore
from tqdm import tqdm

from utils import log_metrics_histograms, log_table, save_csv


def eval_model(
    run, model, dataloader, tokenizer, generate_config, log_config, compute_metrics=True
):
    """
    Evaluates the model by generating responses for the input data and calculates evaluation metrics.

    :param run: wandb run object to log the metrics and results.
    :param model: The language model to be evaluated.
    :param dataloader: Dataloader object that provides batches of data for evaluation.
    :param tokenizer: Tokenizer object to decode the generated responses.
    :param generate_config: Configuration dict for the model's generate function.
    :param log_config: Configuration dict specifying the frequency of logging and the location to save the logs.
    :param compute_metrics: Boolean indicating whether to compute evaluation metrics (default: True).

    :return: None. This function does not return any value. The results are logged using wandb and saved as CSV files.
    """
    model.eval()

    rouge = ROUGEScore()
    bleu = SacreBLEUScore(1, lowercase=True)

    results = []
    metrics_accumulator = {key: [] for key in ["ROUGE_1", "ROUGE_2", "ROUGE_L", "BLEU"]}

    for i, batch in enumerate(tqdm(dataloader)):
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **generate_config,
            ).cpu().numpy()

        prompt_len = len(
            tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)[0]
        )

        gen_answer = str(tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0])
        gen_answer = str(gen_answer[prompt_len:])

        result = {
            "Question": batch["Question"],
            "Answer": batch["Answer"],
            "Generated Answer": gen_answer,
        }

        if compute_metrics:
            rouge_score = rouge(gen_answer, batch["Answer"])
            bleu_score = bleu(gen_answer, batch["Answer"]).item()

            metrics = {
                "ROUGE_1": rouge_score["rouge1_fmeasure"].item(),
                "ROUGE_2": rouge_score["rouge2_fmeasure"].item(),
                "ROUGE_L": rouge_score["rougeL_fmeasure"].item(),
                "BLEU": bleu_score,
            }

            result.update(metrics)

            # accumulate metrics for histogram
            for key in metrics.keys():
                metrics_accumulator[key].append(metrics[key])

        del output_tokens
        torch.cuda.empty_cache()
        gc.collect()

        results.append(result)

        del output_tokens, result
        torch.cuda.empty_cache()
        gc.collect()

        if (i + 1) % log_config["save_steps"] == 0:
            results_df = pd.DataFrame(results)
            save_csv(results_df, f"{log_config['dir']}/{log_config['file_name']}")
            log_table(run, log_config["file_name"], results_df)

            if compute_metrics:
                metrics_df = pd.DataFrame(metrics_accumulator)
                log_metrics_histograms(run, log_config["file_name"], metrics_df)

            # Clear the results for the next iteration
            results = []
            del results_df
            del metrics_df
            gc.collect()

    if results:
        results_df = pd.DataFrame(results)
        save_csv(results_df, f"{log_config['dir']}/{log_config['file_name']}")
        log_table(run, log_config["file_name"], results_df)
        if compute_metrics:
            metrics_df = pd.DataFrame(metrics_accumulator)
            log_metrics_histograms(run, log_config["file_name"], metrics_df)

    artifact = wandb.Artifact(log_config['file_name'].replace('.csv', ''), type='dataset')
    artifact.add_file(f"{log_config['dir']}/{log_config['file_name']}")
    run.log_artifact(artifact)
