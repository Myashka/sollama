from datasets import load_dataset


def make_inference_dataset(
    dataset_name,
    tokenizer,
    split,
    max_prompt_length,
    truncate_promt=True,
    use_title=False,
    columns_to_save=["Question", "Answer"],
    from_file=False,
    **kwargs,
):
    """
    Prepares a dataset for inference by formatting the prompt and tokenizing the text.

    :param dataset_name: The name of the dataset to be loaded for inference.
    :param tokenizer: The tokenizer used to encode the text data.
    :param split: Specifies the portion of the dataset to load (e.g., 'train', 'test').
    :param max_prompt_length: The maximum length of the prompt. If truncate_promt is True, the prompt will be truncated to this length.
    :param truncate_promt: A boolean indicating whether to truncate the prompt to max_prompt_length (default: True).

    :return: Returns a dataset object with the formatted and tokenized text data.
    """

    def _prepare_prompt(question, title=None):
        if title:
            return f"Title: {title}\nQuestion: {question}\n\nAnswer:"
        return f"Question: {question}\n\nAnswer:"

    def promt_tokenize(example):
        if truncate_promt:
            encoded_question = tokenizer.encode(
                example["Question"], add_special_tokens=False
            )
            if use_title:
                encoded_title = tokenizer.encode(
                    "Title: " + example["Title"] + "\nQuestion: \n\nAnswer:",
                    add_special_tokens=False,
                )
                encoded_question = encoded_question[
                    : max_prompt_length - len(encoded_title)
                ]
            else:
                encoded_question = encoded_question[: max_prompt_length - 7]
            tmp = tokenizer.decode(encoded_question, skip_special_tokens=True)
        else:
            tmp = example["Question"]

        if use_title:
            tmp = _prepare_prompt(tmp, example["Title"])
        else:
            tmp = _prepare_prompt(tmp)

        tokenized_dict = tokenizer(
            tmp, padding="longest", max_length=max_prompt_length, truncation=False
        )

        return tokenized_dict

    # if kwargs.get("eval_split"):
    #     split = kwargs["eval_split"]

    if from_file:
        dataset = load_dataset("json", data_files=dataset_name)["train"]
    else:
        dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.map(promt_tokenize)

    dataset.set_format(
        type="torch", columns=columns_to_save + ["input_ids"] + ["attention_mask"]
    )
    return dataset
