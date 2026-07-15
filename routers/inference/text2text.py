from modules.inference.text2text import Text2TextInference
from schemas.inference.text2text import LoadModelArgs, InferenceArgs, ServiceInformation
from schemas import BasicResponse

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(
    prefix= "/v1/text-to-text",
    tags= ["Inference: Text to Text "]
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
    """
    code, outline, detail = await Text2TextInference.load_model(request)
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
    code, outline, detail = await Text2TextInference.unload_model()
    return JSONResponse(BasicResponse(outline= outline, detail= detail).model_dump(), status_code= code)

@router.post(
    "/debug",
    responses= {
        200: {"description": "Success"},
        500: {"description": "Unload Model Error", "model": BasicResponse}
    }
)
async def debug_send_message(body: InferenceArgs):
    """DEBUG"""
    code, outlineOrStreamer, detail = await Text2TextInference.debug_send_request(params= body)
    if code!= 200:
         return JSONResponse(BasicResponse(outline= outlineOrStreamer, detail= detail).model_dump(), status_code= code)
    else:
        # outline is Streamer
        return StreamingResponse(content= outlineOrStreamer, media_type= "text/event-stream")
    
@router.get(
    "/",
    responses= {
        200: {"description": "Return Service Informations.", "model": ServiceInformation},
        400: {"description": "Request Error", "model": BasicResponse}
    }
)
def handle_get_service_info():
    """If Service Running. Return host, port and modelTag."""
    if Text2TextInference.engine.service is not None:
        return JSONResponse(ServiceInformation(host= Text2TextInference.engine.host, port= Text2TextInference.engine.port, modelTag= Text2TextInference.engine.modelTag).model_dump(), status_code= 200)
    return JSONResponse(BasicResponse(outline= "Request Error", detail= "Service is not running.").model_dump(), status_code= 400)