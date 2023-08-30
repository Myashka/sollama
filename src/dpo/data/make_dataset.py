from datasets import load_dataset
from datasets.combine import concatenate_datasets


def print_statistics(dataset, name="Dataset"):
    """Print statistics about the dataset."""
    total = len(dataset)

    gen_k_count = par_k_only_count = par_j_only_count = par_both_count = so_k_count = 0

    for x in dataset:
        if x["is_gen_k"]:
            gen_k_count += 1
        if x["is_par_k"] and not x["is_par_j"]:
            par_k_only_count += 1
        if not x["is_par_k"] and x["is_par_j"]:
            par_j_only_count += 1
        if x["is_par_k"] and x["is_par_j"]:
            par_both_count += 1
        if not (x["is_gen_k"] or x["is_par_k"]):
            so_k_count += 1

    par_count = par_k_only_count + par_j_only_count + par_both_count
    so_j_count = total - par_j_only_count - par_both_count

    print(f"Statistics for {name}:")
    print(f"Total: {total}")
    print(f"Generated (k): {gen_k_count} ({100 * gen_k_count / total:.2f}%)")
    print(
        f"Paraphrased (k only): {par_k_only_count} ({100 * par_k_only_count / total:.2f}%)"
    )
    print(
        f"Paraphrased (j only): {par_j_only_count} ({100 * par_j_only_count / total:.2f}%)"
    )
    print(
        f"Paraphrased (both j & k): {par_both_count} ({100 * par_both_count / total:.2f}%)"
    )
    print(f"SO (k): {so_k_count} ({100 * so_k_count / total:.2f}%)")
    print(f"SO (j): {so_j_count} ({100 * so_j_count / total:.2f}%)")
    print(f"Total Paraphrased: {par_count} ({100 * par_count / total:.2f}%)")
    print(
        f"Total SO: {total - gen_k_count - par_count} ({100 * (total - gen_k_count - par_count) / total:.2f}%)"
    )
    print("-" * 50)


def make_datasets(
    dataset_name,
    tokenizer,
    max_prompt_length,
    max_answer_length,
    use_title=False,
    test_size=1000,
    val_size=500,
    use_gen=False,
    use_par=False,
    use_so=True,
    do_train=True,
    num_proc=8,
    so_fix=None,
):
    def _prepare_prompt(question, title=None):
        if title:
            return f"Title: {title}\nQuestion: {question}\n\nAnswer:"
        return f"Question: {question}\n\nAnswer:"
    
    def truncate_question(question, title, max_length):
        prompt_skeleton = "Title: {}\nQuestion: {}\n\nAnswer:" if title else "Question: {}\n\nAnswer:"
        prompt_length = len(prompt_skeleton.format(title if title else "", "", ""))
        truncated_question = question[:max_length - prompt_length]
        return truncated_question

    def prompt_tokenize(example):
        question = example["Question"]
        title = example["Title"] if use_title else None
        truncated_question = truncate_question(question, title, max_prompt_length)

        chosen = example["response_j"]
        rejected = example["response_k"]

        chosen_encoded = tokenizer.encode(chosen, max_length=max_answer_length, truncation=True)
        rejected_encoded = tokenizer.encode(rejected, max_length=max_answer_length, truncation=True)

        chosen_text = tokenizer.decode(chosen_encoded, skip_special_tokens=True)
        rejected_text = tokenizer.decode(rejected_encoded, skip_special_tokens=True)

        prompt = _prepare_prompt(truncated_question, title)
        return {"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text}

    def filter_datasets(example):
        if use_gen and (example["is_gen_k"] or example["is_gen_j"]):
            return True
        if use_par and (example["is_par_k"] or example["is_par_j"]):
            return True
        if use_so and not (example["is_gen_k"] or example["is_gen_j"] or example["is_par_k"] or example["is_par_j"]):
            return True
        return False
        
    if not use_gen and not use_par:
        so_fix = None

    dataset = load_dataset(dataset_name)["train"]
    dataset = load_dataset(dataset_name)["train"].filter(filter_datasets, num_proc=num_proc)
    dataset = dataset.map(prompt_tokenize, num_proc=num_proc)

    if not so_fix:
        train_test_split = dataset.train_test_split(test_size=test_size)
        temp_train_data = train_test_split["train"]
        test_data = train_test_split["test"]

        train_val_split = temp_train_data.train_test_split(test_size=val_size)
        train_data = train_val_split["train"]
        val_data = train_val_split["test"]

        if do_train:
            print_statistics(train_data, "Train Dataset")
            print_statistics(val_data, "Validation Dataset")

            col_to_remove = [col for col in train_data.column_names if col not in ['prompt', 'chosen', 'rejected']]
            train_data = train_data.remove_columns(col_to_remove)
            val_data = val_data.remove_columns(col_to_remove)

            return train_data, val_data

        print_statistics(test_data, "Test Dataset")
        col_to_remove = [col for col in test_data.column_names if col not in ['prompt', 'chosen', 'rejected']]
        test_data = test_data.remove_columns(col_to_remove)
        return test_data
    else:
        test_so_part = int(test_size * so_fix)
        test_any_part = test_size - test_so_part

        val_so_part = int(val_size * so_fix)
        val_any_part = val_size - val_so_part

        any_dataset = dataset.filter(
            lambda x: (
                x["is_gen_k"] or x["is_gen_j"] or x["is_par_k"] or x["is_par_j"]
            ),
            num_proc=num_proc,
        )
        so_dataset = dataset.filter(
            lambda x: not (
                x["is_gen_k"] or x["is_gen_j"] or x["is_par_k"] or x["is_par_j"]
            ),
            num_proc=num_proc,
        )

        test_so_split = so_dataset.train_test_split(test_size=test_so_part)
        test_any_split = any_dataset.train_test_split(test_size=test_any_part)

        if not do_train:
            test_data = concatenate_datasets(
                [test_so_split["test"], test_any_split["test"]]
            )
            print_statistics(test_data, "Test Dataset")
            col_to_remove = [col for col in test_data.column_names if col not in ['prompt', 'chosen', 'rejected']]
            test_data = test_data.remove_columns(col_to_remove)
            return test_data

        train_val_so_split = test_so_split["train"].train_test_split(
            test_size=val_so_part
        )
        train_val_any_split = test_any_split["train"].train_test_split(
            test_size=val_any_part
        )

        train_data = concatenate_datasets(
            [train_val_so_split["train"], train_val_any_split["train"]]
        )
        val_data = concatenate_datasets(
            [train_val_so_split["test"], train_val_any_split["test"]]
        )

        print_statistics(train_data, "Train Dataset")
        print_statistics(val_data, "Validation Dataset")

        col_to_remove = [col for col in train_data.column_names if col not in ['prompt', 'chosen', 'rejected']]
        train_data = train_data.remove_columns(col_to_remove)
        val_data = val_data.remove_columns(col_to_remove)

        return train_data, val_data