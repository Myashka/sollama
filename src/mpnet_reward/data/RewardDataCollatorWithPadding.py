from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, List, Optional, Union
from transformers.utils import PaddingStrategy
from dataclasses import dataclass
import torch

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_a = []
        features_p = []
        features_n = []
        is_gen_k = []
        is_par_j = []
        is_par_k = []
        for feature in features:
            is_gen_k.append(feature["is_gen_k"])
            is_par_j.append(feature["is_par_j"])
            is_par_k.append(feature["is_par_k"])
            features_a.append(
                {
                    "input_ids": feature["input_ids_a"],
                    "attention_mask": feature["attention_mask_a"],
                }
            )
            features_p.append(
                {
                    "input_ids": feature["input_ids_p"],
                    "attention_mask": feature["attention_mask_p"],
                }
            )
            features_n.append(
                {
                    "input_ids": feature["input_ids_n"],
                    "attention_mask": feature["attention_mask_n"],
                }
            )
        batch_a = self.tokenizer.pad(
            features_a,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_p = self.tokenizer.pad(
            features_p,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_n = self.tokenizer.pad(
            features_n,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_a": batch_a["input_ids"],
            "attention_mask_a": batch_a["attention_mask"],
            "input_ids_p": batch_p["input_ids"],
            "attention_mask_p": batch_p["attention_mask"],
            "input_ids_n": batch_n["input_ids"],
            "attention_mask_n": batch_n["attention_mask"],
            "is_gen_k": torch.tensor(is_gen_k),
            "is_par_j": torch.tensor(is_par_j),
            "is_par_k": torch.tensor(is_par_k),
            "return_loss": True,
        }
        return batch