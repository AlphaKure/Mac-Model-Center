import asyncio

from schemas.inference.text2image import LoadModelArgs, RequestArgs
from modules.inference.text2image.core import Text2Image

class Text2ImageInterface:

    core = Text2Image()
    lock = asyncio.Lock() # Avoid load model and unload model at same time
    _service : asyncio.Task| None = None 

    @classmethod
    async def load_model(cls, loadModelArgs: LoadModelArgs):
        """ For Text2Image Model Load"""
        """ Return status code, outline, detail """
        
        if cls.core.isLoading or cls.core.model is not None:
            return 400, "Request Error", "Already load model or loading "

        async with cls.lock:
            
            try:
                await asyncio.to_thread(cls.core.load_model, loadModelArgs= loadModelArgs)
            except Exception as error:
                return 500, "Model Load Error", str(error)  

            try:
                cls._service = asyncio.create_task(await asyncio.to_thread(cls.core._service))
            except Exception as error:
                return 500, "Startup Service Error", str(error)

            return 200, "Success", "Success"

    @classmethod
    async def unload_model(cls):
        """ For Text2Image Model Unload"""
        """ Return status code, outline, detail """
        
        if cls.core.isLoading or cls.core.model is None:
            return 400, "Request Error", "Didn't load model or loading "
    
        async with cls.lock:

            try:
                cls.core.unload_model()
            except Exception as error:
                return 500, "Unload Model Error", str(error)
        
            return 200, "Success", "Success"
    
    @classmethod
    async def interface(cls, requestArg: RequestArgs):
        """For Text2Image Model Interface"""
        """
        if Success:
            Return status code, Generate, None 
        if Fail:
            Return status code, outline, detail 
        """

        if cls.core.isLoading or cls.core.model is None:
            return 400, "Request Error", "Didn't load model or loading "

        try:
            stream = await cls.core.interface(requestArg)        
        except Exception as error:
            return 500, "Inference Error", str(error)
        return 200, stream, None

    @classmethod
    async def cancel(cls, requestID: str|None):
        """Interrupt Request"""

        if cls.core.isLoading or cls.core.model is None:
            return 400, "Request Error", "Didn't load model or loading "
        
        try:
            cls.core.cancel(requestID)
        except Exception as error:
            return 500, "Cancel Error", str(error)
        return 200, "Success", "Success"
