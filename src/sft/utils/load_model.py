from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel



def load_model(model_config):
    tokenizer = LlamaTokenizer.from_pretrained(model_config["name"], eos_token="</s>")
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = LlamaForCausalLM.from_pretrained(
        model_config["name"],
        load_in_8bit=model_config["load_in_8bit"],
        device_map="auto",
    )
    if model_config.get('peft_model_id'):
        model = PeftModel.from_pretrained(model, model_config['peft_model_id'])


    return model, tokenizer
