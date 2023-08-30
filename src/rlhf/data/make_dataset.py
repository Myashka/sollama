from datasets import load_dataset
import numpy as np

def make_dataset(
    dataset_name,
    tokenizer,
    split,
    max_prompt_length,
    truncate_promt=True,
    use_title=False,
    from_file=False,
    **kwargs,
):
    def _prepare_prompt(question, title=None):
        if title:
            return f"Title: {title}\nQuestion: {question}\n\nAnswer:"
        return f"Question: {question}\n\nAnswer:"

    def promt_tokenize(example):
        if truncate_promt:
            encoded_question = tokenizer.encode(example["Question"], add_special_tokens=False)
            if use_title:
                encoded_title = tokenizer.encode(
                    "Title: " + example["Title"] + "\nQuestion: \n\nAnswer:",
                    add_special_tokens=False,
                )
                encoded_question = encoded_question[: max_prompt_length - len(encoded_title)]
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

        tokenized_dict["query"] = tmp

        return tokenized_dict

    # if kwargs.get("eval_split"):
    #     split = kwargs["eval_split"]

    if from_file:
        dataset = load_dataset("json", data_files=dataset_name)["train"]
    else:
        dataset = load_dataset(dataset_name, split=split)
        unique_ids = np.unique(dataset['Q_Id'])
        dataset = dataset.filter(lambda example: example['Q_Id'] in unique_ids)
    dataset = dataset.map(promt_tokenize, num_proc=8)

    dataset.set_format(type="torch", columns=["input_ids", "query"])
    return dataset


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
