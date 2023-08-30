import gc
import os
import pandas as pd
import torch
import wandb
from tqdm import tqdm
import torch.nn.functional as F

from src.mpnet_reward.utils import log_table, save_csv
from .helper_fucntions import get_embeddings_for_batch, compute_embeddings_sim

def log_results(results, log_config, run):
    os.makedirs(log_config["dir"], exist_ok=True)

    save_csv(results, f"{log_config['dir']}/{log_config['file_name']}")
    log_table(run, log_config["file_name"], results)


def eval_mpnet_model(
    run,
    model,
    dataloader_ids,
    dataloader_qa,
    log_config,
    columns_to_save,
):
    results = pd.DataFrame()

    for i, (batch_ids, batch_qa) in enumerate(
        tqdm(zip(dataloader_ids, dataloader_qa), total=len(dataloader_qa))
    ):
        batch_ids.pop('return_loss', None)
        batch_ids = {k: v.to(model.device) for k, v in batch_ids.items()}
        embeddings_q, embeddings_j, embeddings_k = get_embeddings_for_batch(model, batch_ids)

        j_sim_dict = compute_embeddings_sim(embeddings_q, embeddings_j, '_j')
        k_sim_dict = compute_embeddings_sim(embeddings_q, embeddings_k, '_k')
        jk_sim_dict = compute_embeddings_sim(embeddings_j, embeddings_k, '_jk')

        result_dict = {col: batch_qa[col] for col in columns_to_save}
        result_dict.update(j_sim_dict)
        result_dict.update(k_sim_dict)
        result_dict.update(jk_sim_dict)

        result = pd.DataFrame(result_dict)

        results = pd.concat([results, result], ignore_index=True)

        del embeddings_q, embeddings_j, embeddings_k, result, j_sim_dict, k_sim_dict
        torch.cuda.empty_cache()
        gc.collect()

        if (i + 1) % log_config["save_steps"] == 0:
            log_results(results, log_config, run)

            # Clear the results for the next iteration
            results = pd.DataFrame()
            gc.collect()

    if not results.empty:
        log_results(results, log_config, run)

    artifact = wandb.Artifact(
        log_config["file_name"].replace(".csv", ""), type="dataset"
    )
    artifact.add_file(f"{log_config['dir']}/{log_config['file_name']}")
    run.log_artifact(artifact)
