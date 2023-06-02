from transformers import LlamaForCausalLM, LlamaTokenizer


def load_model(model_config):
    """
    Loads a pretrained language model and its corresponding tokenizer based on the provided configuration.

    :param model_config: Configuration dict specifying the model's name and whether to load the model in 8-bit mode.

    :return: tuple, (model, tokenizer). The pretrained language model and its corresponding tokenizer.
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_config["name"], eos_token="</s>")
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = LlamaForCausalLM.from_pretrained(
        model_config["name"],
        load_in_8bit=model_config["load_in_8bit"],
        device_map="auto",
    )

    return model, tokenizer
