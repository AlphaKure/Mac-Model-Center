from modules.inference.text2image import Text2ImageInterface
from schemas.inference.text2image import RequestArgs, LoadModelArgs
from schemas.inference import ProgressResponse
from schemas import BasicResponse

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(
    prefix= "/v1/text-to-image",
    tags= ["Inference: Text to Image "]
)

@router.post(
    "/",
    responses= {
        200: {"description": "Success", "model": BasicResponse},
        400: {"description": "Request Error", "model": BasicResponse},
        500: {"description": "Model Load Error or Startup Service Error", "model": BasicResponse}
    }
    )
async def handle_load_model(request: LoadModelArgs):
    """
    # Load model
    ## Request Body
        - model_path (str): Use model local path.
        - dtype (str): Now only support "float16" or "bfloat16".
    """
    code, outline, detail = await Text2ImageInterface.load_model(request)
    return JSONResponse(BasicResponse(outline= outline, detail= detail).model_dump(), status_code= code)

@router.delete(
    "/",
    responses= {
        200: {"description": "Success", "model": BasicResponse},
        400: {"description": "Request Error", "model": BasicResponse},
        500: {"description": "Unload Model Error", "model": BasicResponse}
    }
)
async def handle_unload_model():
    """
    # Unload model
    """
    code, outline, detail = await Text2ImageInterface.unload_model()
    return JSONResponse(BasicResponse(outline= outline, detail= detail).model_dump(), status_code= code)

@router.post(
    "/inference",
    responses = {
        200: {"description": "Success(Streaming)", "model": ProgressResponse},
        400: {"description": "Request Error", "model": BasicResponse},
        500: {"description": "Unload Model Error", "model": BasicResponse}
    }
)
async def handle_inference_interface(request: RequestArgs):
    """
    # Inference Interface
    ## RequestBody
        - prompt (str): User prompt.
        - width (int): Output picture width.
        - height (int): Output picture height. 
        - num_inference_steps (int): Generate steps.
        - guidance_scale (float): Generate scale.
        - output_path (str): Output picture path.
        - seed (Optional[int]): Generate seed.

    ## Tip
        * Success response is SSE streaming response
    """
    code, outline, detail = await Text2ImageInterface.interface(request)
    if code == 200:
        return StreamingResponse(outline, media_type="text/event-stream", status_code= 200)
    else:
        return JSONResponse(BasicResponse(outline= outline, detail= detail).model_dump(), status_code= code)

@router.delete(
    "/inference",
    responses = {
        200: {"description": "Success", "model": BasicResponse},
        400: {"description": "Request Error", "model": BasicResponse},
        500: {"description": "Cancel Error", "model": BasicResponse}
    }
)
async def handle_inference_interrupt(id: str|None= None):
    """
    # Interrupt Generate
    ## Query:
        - id (Optional[str]): Stop generate. If not give id. stop all of tasks.
    """
    code, outline, detail = await Text2ImageInterface.cancel(requestID= id)
    return JSONResponse(BasicResponse(outline= outline, detail= detail).model_dump(), status_code= code)