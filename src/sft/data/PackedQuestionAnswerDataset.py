from typing import Dict, List, Union
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch


class PackedQuestionAnswerDataset(Dataset):
    """
    A class that extends torch.utils.data.Dataset to create a dataset
    for a question answering task with packed sequences.

    :param dataset: The list of dictionaries each containing a question and answer pair.
    :param tokenizer: Tokenizer instance to encode the texts.
    :param max_length: The maximum length for the sequences.
    :param max_prompt_length: The maximum length for the prompt sequences.
    :param use_title: A flag indicating whether to include titles in the prompt.
    :return: An instance of PackedQuestionAnswerDataset.
    """

    def __init__(
        self,
        dataset: List[Dict[str, Union[str, int]]],
        tokenizer,
        max_length: int,
        max_prompt_length: int,
        use_title: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.use_title = use_title
        self.packed_samples = self.pack_inputs_and_labels(dataset)

    def prepare_prompt(self, question: str, title: str = None) -> str:
        if title:
            return f"Title: {title}\nQuestion: {question}"
        return f"Question: {question}"

    def encode_example(self, example: Dict[str, str]) -> Dict[str, List[int]]:
        """
        Encode a single question and answer pair.

        :param example: A dictionary containing the question and answer.
        :return: A dictionary with encoded question and answer.
        """
        question = (
            self.prepare_prompt(example["Question"], example["Title"])
            if self.use_title
            else self.prepare_prompt(example["Question"])
        )
        answer = "\n\nAnswer: " + example["Answer"]
        answer += "</s>\n\n"

        encoded_answer = self.tokenizer.encode(answer, add_special_tokens=False)
        encoded_question = self.tokenizer.encode(question, add_special_tokens=False)

        encoded_question = encoded_question[: self.max_prompt_length]

        return {"question": encoded_question, "answer": encoded_answer}

    def build_sequence(
        self, question: List[int], answer: List[int]
    ) -> Dict[str, List[int]]:
        """
        Build the input and label sequences for a question-answer pair.

        :param question: The encoded question.
        :param answer: The encoded answer.
        :return: A dictionary containing the input and label sequences.
        """
        input_sequence = question + answer
        label_sequence = [-100] * len(question) + answer

        input_sequence = input_sequence[: self.max_length]
        label_sequence = label_sequence[: self.max_length]

        return {"input": input_sequence, "label": label_sequence}

    def build_sample(self, buffer: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
        """
        Build a sample from input and label buffers.

        :param buffer: A dictionary containing the input and label buffers.
        :return: A dictionary with tensors for input_ids, labels, and attention_mask.
        """
        return {
            "input_ids": torch.tensor(buffer["input"]),
            "attention_mask": torch.ones(len(buffer["input"]), dtype=int),
            "labels": torch.tensor(buffer["label"]),
        }

    def pack_inputs_and_labels(self, dataset):
        """
        Pack the inputs and labels for all examples in the dataset.

        :param dataset: The list of dictionaries each containing a question and answer pair.
        :return: A list of samples each containing tensors for input_ids, labels, and attention_mask.
        """
        packed_samples = []
        input_buffer, label_buffer = [], []
        for example in tqdm(dataset):
            encoded_example = self.encode_example(example)
            sequence = self.build_sequence(
                encoded_example["question"], encoded_example["answer"]
            )

            if len(input_buffer) + len(sequence["input"]) > self.max_length:
                packed_samples.append(
                    self.build_sample({"input": input_buffer, "label": label_buffer})
                )
                input_buffer, label_buffer = sequence["input"], sequence["label"]
            else:
                input_buffer += sequence["input"]
                label_buffer += sequence["label"]

        if input_buffer and len(input_buffer) <= self.max_length:
            packed_samples.append(
                self.build_sample({"input": input_buffer, "label": label_buffer})
            )

        return packed_samples

    def __len__(self) -> int:
        return len(self.packed_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.packed_samples[idx]
