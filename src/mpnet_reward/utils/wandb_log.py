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