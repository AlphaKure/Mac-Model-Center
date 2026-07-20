import asyncio 

from schemas.convert import ConvertArgForMLX
from modules.convert.mlx import llm_quantize_to_mlx

async def mlx_llm_quant(
    args: ConvertArgForMLX
):
    try:
        await asyncio.to_thread(llm_quantize_to_mlx, args= args)
        return 200, "Success", "Success"
    except Exception as error:
        return 500, "Convert Fail", str(error)