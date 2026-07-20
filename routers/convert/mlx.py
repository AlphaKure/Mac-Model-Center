from modules.convert import mlx_llm_quant
from schemas.convert import ConvertArgForMLX
from schemas import BasicResponse

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(
    prefix= "/v1/convert",
    tags= ["Convert: MLX"]
)

@router.post(
    "/llm",
    responses= {
        200: {"description": "Success", "model": BasicResponse},
        500: {"description": "Convert Error", "model": BasicResponse}
    }
    )
async def handle_load_model(request: ConvertArgForMLX):
    """
    # Load model
    ## Request Body
        - model_path (str): Use model local path.
        - outputPath (str): Output path. 
        - precision (Literal["bfloat16", "int8", "int4", "mxfp4", "mxfp8", "nvfp4"]): Only support below precision.
        - dtype (str): Original model dtype.
    """
    code, outline, detail = await mlx_llm_quant(request)
    return JSONResponse(BasicResponse(outline= outline, detail= detail).model_dump(), status_code= code)