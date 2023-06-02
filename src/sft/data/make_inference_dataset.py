from datasets import load_dataset


def make_inference_dataset(
    dataset_name,
    tokenizer,
    split,
    max_prompt_length,
    padding="longest",
    truncate_promt=True,
):
    """
    Prepares a dataset for inference by formatting the prompt and tokenizing the text.

    :param dataset_name: The name of the dataset to be loaded for inference.
    :param tokenizer: The tokenizer used to encode the text data.
    :param split: Specifies the portion of the dataset to load (e.g., 'train', 'test').
    :param max_prompt_length: The maximum length of the prompt. If truncate_promt is True, the prompt will be truncated to this length.
    :param padding: The type of padding to be applied after tokenization (default: "longest").
                    Can be one of ['longest', 'max_length', 'do_not_pad']
    :param truncate_promt: A boolean indicating whether to truncate the prompt to max_prompt_length (default: True).

    :return: Returns a dataset object with the formatted and tokenized text data.
    """

    def _prepare_prompt(question):
        return f"Question: {question}\nAnswer:"

    def promt_tokenize(examples):
        if truncate_promt:
            q_toks = tokenizer.encode(examples["Question"])
            q_toks = q_toks[: max_prompt_length - 8]
            tmp = tokenizer.decode(q_toks).strip()
        else:
            tmp = examples["Question"]

        tmp = _prepare_prompt(tmp)

        tokenized_dict = tokenizer(
            tmp, padding=padding, max_length=max_prompt_length, truncation=True
        )

        return tokenized_dict

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.map(promt_tokenize)
    dataset.set_format(
        type="torch", columns=["Question", "Answer", "input_ids", "attention_mask"]
    )

    return dataset
