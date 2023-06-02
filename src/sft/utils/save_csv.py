import os


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
