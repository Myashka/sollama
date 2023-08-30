from datasets import load_dataset
from .PackedQuestionAnswerDataset import PackedQuestionAnswerDataset


def make_train_dataset(
    dataset_name,
    tokenizer,
    split,
    max_length,
    max_prompt_length,
    use_title=False,
    **kwargs
):
    dataset = load_dataset(dataset_name, split=split)
    if split=='train' and kwargs.get("filter_zero_scores"):
        dataset = dataset.filter(lambda example: example["Score"] > 0)
        # dataset = dataset.filter(lambda example: example["is_accepted"] == True)
    dataset = dataset.shuffle()
    dataset = PackedQuestionAnswerDataset(
        dataset, tokenizer, max_length, max_prompt_length, use_title
    )

    return dataset
