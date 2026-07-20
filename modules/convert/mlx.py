from typing import Literal

from mlx_lm import convert


def quantize_llm_to_mlx(
        modelPath: str,
        outputPath: str,
        precision: Literal["bfloat16", "int8", "int4", "mxfp4", "mxfp8", "nvfp4"],
        dtype: str = "bfloat16"
):

    match precision:
        
        case "bfloat16":
            quantize = False
            qBit = None
            qMode = None
        
        case "int8":
            quantize = True
            qBit = 8
            qMode = "affine"
        
        case "int4":
            quantize = True
            qBit = 4
            qMode = "affine"
        
        case "mxfp4":
            quantize = True
            qBit = None
            qMode = "mxfp4"
        
        case "nvfp4":
            quantize = True
            qBit = None
            qMode = "nvfp4"
        
        case "mxfp8":
            quantize = True
            qBit = None
            qMode = "mxfp8"

        case _:
            raise IndexError("Invalid precision.")


    convert(
        hf_path= modelPath,
        mlx_path= outputPath,
        quantize= quantize,
        q_bits= qBit,
        q_mode= qMode, # type: ignore
        dtype= dtype
    )

if __name__ == "__main__":
    quantize_llm_to_mlx(
        modelPath= "/Users/alpha/model/Qwen3-4B-Thinking-2507",
        outputPath= "/Users/alpha/model/Qwen3-4B-Thinking-2507-mlx-8bits",
        precision= "int8"
    )