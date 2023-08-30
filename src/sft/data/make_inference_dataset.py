from datasets import load_dataset


def make_inference_dataset(
    dataset_name,
    tokenizer,
    split,
    max_prompt_length,
    padding="longest",
    truncate_promt=True,
    use_title=False,
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

    def _prepare_prompt(question, title=None):
        if title:
            return f"Title: {title}\nQuestion: {question}\n\nAnswer:"
        return f"Question: {question}\n\nAnswer:"
    
    def promt_tokenize(example):
        if truncate_promt:
            encoded_question = tokenizer.encode(example["Question"], add_special_tokens=False)
            if use_title:
                encoded_title = tokenizer.encode("Title: " + example["Title"], add_special_tokens=False)
                encoded_question = encoded_question[: max_prompt_length - len(encoded_title) - 8]
            else:
                encoded_question = encoded_question[: max_prompt_length - 8]
            tmp = tokenizer.decode(encoded_question, skip_special_tokens=True)
        else:
            tmp = example["Question"]

        if use_title:
            tmp = _prepare_prompt(tmp, example['Title'])
        else:
            tmp = _prepare_prompt(tmp)

        tokenized_dict = tokenizer(
            tmp, padding=padding, max_length=max_prompt_length, truncation=True
        )

        return tokenized_dict

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.map(promt_tokenize)
    if use_title:
        dataset.set_format(
            type="torch",
            columns=["Title", "Question", "Answer", "input_ids", "attention_mask"],
        )
    else:
        dataset.set_format(
            type="torch", columns=["Question", "Answer", "input_ids", "attention_mask"]
        )
    return dataset
