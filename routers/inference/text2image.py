from modules.inference.text2image import Text2ImageInterface
from schemas.inference.text2image import RequestArgs, LoadModelArgs
from schemas import BasicResponse

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(
    prefix= "/v1/text-to-image",
    tags= ["Inference: Text to Image "]
)

@router.post("/")
async def handle_load_model(request: LoadModelArgs):
    
    code, outline, detail = await Text2ImageInterface.load_model(request)
    return JSONResponse(BasicResponse(outline= outline, detail= detail).model_dump(), status_code= code)

@router.delete("/")
async def handle_unload_model():
    
    code, outline, detail = await Text2ImageInterface.unload_model()
    return JSONResponse(BasicResponse(outline= outline, detail= detail).model_dump(), status_code= code)

@router.post("/inference")
async def handle_inference_interface(request: RequestArgs):
    
    code, outline, detail = await Text2ImageInterface.interface(request)
    if code == 200:
        return StreamingResponse(outline, media_type="text/event-stream", status_code= 200)
    else:
        return JSONResponse(BasicResponse(outline= outline, detail= detail).model_dump(), status_code= code)

@router.delete("/inference")
async def handle_inference_interrupt(id: str|None= None):
    
    code, outline, detail = await Text2ImageInterface.cancel(requestID= id)
    return JSONResponse(BasicResponse(outline= outline, detail= detail).model_dump(), status_code= code)