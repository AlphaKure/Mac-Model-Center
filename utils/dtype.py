import torch

def str_to_dtype(input: str):
    """Change string type to torch dtype"""
    match input:
        case "float16":
            return torch.float16
        case "bfloat16":
            return torch.bfloat16
        case _:
            raise KeyError("Invalid dtype")