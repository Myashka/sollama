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
    max_length_q,
    max_length_a,
    use_title=False,
    test_size=1000,
    val_size=500,
    use_gen=False,
    use_par=False,
    use_so=True,
    do_train=True,
    num_proc=8,
    so_fix = None,
    columns_to_save=[],
):
    def _prepare_prompt(question, title=None):
        if title:
            return f"{title}\n{question}"
        return f"{question}"

    def prompt_tokenize(example):
        question = example["Question"]
        title = example["Title"] if use_title else None
        positive = example["response_j"]
        negative = example["response_k"]

        prompt = _prepare_prompt(question, title)
        anchor_dict = tokenizer(
            prompt, padding="longest", max_length=max_length_q, truncation=True
        )
        positive_dict = tokenizer(
            positive, padding="longest", max_length=max_length_a, truncation=True
        )
        negative_dict = tokenizer(
            negative, padding="longest", max_length=max_length_a, truncation=True
        )

        return {
            "input_ids_a": anchor_dict["input_ids"],
            "attention_mask_a": anchor_dict["attention_mask"],
            "input_ids_p": positive_dict["input_ids"],
            "attention_mask_p": positive_dict["attention_mask"],
            "input_ids_n": negative_dict["input_ids"],
            "attention_mask_n": negative_dict["attention_mask"],
        }

    if not use_gen and not use_par:
        so_fix = None

    dataset = load_dataset(dataset_name)["train"]

    gen_subset = dataset.filter(lambda x: x["is_gen_k"] == True or x["is_gen_j"] == True, num_proc=num_proc)
    par_subset = dataset.filter(lambda x: x["is_par_k"] == True or x["is_par_j"] == True, num_proc=num_proc)
    so_subset = dataset.filter(lambda x: not (x["is_gen_k"] or x["is_gen_j"] or x["is_par_k"] or x["is_par_j"]), num_proc=num_proc)

    datasets_to_concatenate = []

    if use_gen:
        datasets_to_concatenate.append(gen_subset)
    if use_par:
        datasets_to_concatenate.append(par_subset)
    if use_so:
        datasets_to_concatenate.append(so_subset)

    dataset = concatenate_datasets(datasets_to_concatenate)

    dataset = dataset.map(prompt_tokenize, num_proc=num_proc)

    columns = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_p",
        "attention_mask_p",
        "input_ids_n",
        "attention_mask_n",
        "is_gen_k",
        "is_par_j",
        "is_par_k",
    ]

    # dataset.set_format(type="torch", columns=columns+columns_to_save)
    if not so_fix:
        train_test_split = dataset.train_test_split(test_size=test_size)
        temp_train_data = train_test_split["train"]
        test_data = train_test_split["test"]

        train_val_split = temp_train_data.train_test_split(test_size=val_size)
        train_data = train_val_split["train"]
        val_data = train_val_split["test"]

        if do_train:
            train_data.set_format(type="torch", columns=columns+columns_to_save)
            val_data.set_format(type="torch", columns=columns+columns_to_save)

            print_statistics(train_data, "Train Dataset")
            print_statistics(val_data, "Validation Dataset")

            return train_data, val_data

        test_data.set_format(type="torch", columns=columns+columns_to_save)
        print_statistics(test_data, "Test Dataset")
        return test_data
    else:
        test_so_part = int(test_size * so_fix)
        test_any_part = test_size - test_so_part

        val_so_part = int(val_size * so_fix)
        val_any_part = val_size - val_so_part

        any_dataset = dataset.filter(lambda x: (x["is_gen_k"] or x["is_gen_j"] or x["is_par_k"] or x["is_par_j"]), num_proc=num_proc)
        so_dataset = dataset.filter(lambda x: not (x["is_gen_k"] or x["is_gen_j"] or x["is_par_k"] or x["is_par_j"]), num_proc=num_proc)

        test_so_split = so_dataset.train_test_split(test_size=test_so_part)
        test_any_split = any_dataset.train_test_split(test_size=test_any_part)

        if not do_train:
            test_data = concatenate_datasets([test_so_split['test'], test_any_split['test']])
            test_data.set_format(type="torch", columns=columns+columns_to_save)
            print_statistics(test_data, "Test Dataset")
            return test_data
        
        train_val_so_split = test_so_split['train'].train_test_split(test_size=val_so_part)
        train_val_any_split = test_any_split['train'].train_test_split(test_size=val_any_part)

        train_data = concatenate_datasets([train_val_so_split["train"], train_val_any_split["train"]])
        val_data = concatenate_datasets([train_val_so_split["test"], train_val_any_split["test"]])
        train_data.set_format(type="torch", columns=columns+columns_to_save)
        val_data.set_format(type="torch", columns=columns+columns_to_save)

        print_statistics(train_data, "Train Dataset")
        print_statistics(val_data, "Validation Dataset")

        return train_data, val_data