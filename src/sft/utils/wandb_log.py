import wandb


def log_table(run, filename, table_df):
    """
    Logs a table of results to wandb.

    :param run: wandb run object to log the table.
    :param filename: String denoting the name of the file for the log. It will be used as the key in the logged data.
    :param table_df: Pandas DataFrame to be logged as a table.

    :return: None. The function does not return any value. The table is logged using wandb.
    """

    filename = filename.replace('.csv', '')
    wandb_table = wandb.Table(dataframe=table_df)
    run.log({filename: wandb_table})


def log_metrics_histograms(run, filename, table_df):
    """
    Logs histograms of the metrics to wandb.

    :param run: wandb run object to log the histograms.
    :param filename: String denoting the name of the file for the log. It will be used as the key in the logged data.
    :param metrics_df: Pandas DataFrame containing the metrics data to be logged as histograms.

    :return: None. The function does not return any value. The histograms are logged using wandb.
    """
    metrics = ["BLEU", "ROUGE_1", "ROUGE_2", "ROUGE_L"]
    filename = filename.replace('.csv', '')
    for metric in metrics:
        wandb_table = wandb.Table(dataframe=table_df[[metric]])
        run.log(
            {
                f"{filename}_{metric}": wandb.plot.histogram(
                    wandb_table, metric, title=f"{metric} Histogram"
                )
            }
        )
