from schemas.convert import ConvertArgForMLX

from mlx_lm import convert


def llm_quantize_to_mlx(
    args: ConvertArgForMLX,
):

    match args.precision:
        
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
        hf_path= args.modelPath,
        mlx_path= args.outputPath,
        quantize= quantize,
        q_bits= qBit,
        q_mode= qMode, # type: ignore
        dtype= args.dtype
    )
