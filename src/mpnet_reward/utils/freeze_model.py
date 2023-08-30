import torch

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def freeze_model(model, parameters_to_modify: dict):
    """
    Freeze all parameters of the given model except those specified in parameters_to_modify.

    :param model: The PyTorch model to be frozen.
    :type model: torch.nn.Module
    :param parameters_to_modify: Dictionary specifying the parameters to be unfrozen. The keys should correspond to the attributes' paths in the model, and the values should be set to True to unfreeze the corresponding parameters.
    :type parameters_to_modify: dict

    Example:
        parameters_to_modify = {
            "embeddings": True,
            "encoder.layer.1": True,
            "encoder.layer.11": True,
        }
        freeze_model(model, parameters_to_modify)
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    for key, value in parameters_to_modify.items():
        if value:
            parts = key.split(".")
            target = model
            for part in parts:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    raise KeyError(f"Unable to find attribute {key} in model.")

            if isinstance(target, torch.nn.Parameter):
                target.requires_grad = True

            elif isinstance(target, torch.nn.Module):
                for param in target.parameters():
                    param.requires_grad = True

    print_trainable_parameters(model)