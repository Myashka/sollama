from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch


def load_model(model_config):
    tokenizer = LlamaTokenizer.from_pretrained(model_config["name"], eos_token="</s>")
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = model_config['padding_side']

    if model_config["torch_dtype"] == 'fp16':
        torch_dtype = torch.float16
    else:
        torch_dtype = None


    model = LlamaForCausalLM.from_pretrained(
        model_config["name"],
        load_in_8bit=model_config["load_in_8bit"],
        torch_dtype=torch_dtype,
        device_map=model_config["device_map"],
    )
    if model_config.get("peft_model_id"):
        model = PeftModel.from_pretrained(model, model_config["peft_model_id"])

    return model, tokenizer
