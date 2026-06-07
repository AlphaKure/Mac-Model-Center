import asyncio 
from datetime import datetime
from functools import partial
import gc
import json
import os
from typing import Literal
from uuid import uuid4


from utils.dtype import str_to_dtype
from schemas.inference.text2image import RequestArgs, LoadModelArgs
from schemas.inference import Progress
from modules.inference import _callback

from diffusers import AutoPipelineForText2Image
import torch

class Text2Image:

    model = None
    isLoading = False
    communicationQueue: dict[str, asyncio.Queue] = {} # Use for send process bar
    requestQueue: asyncio.Queue = asyncio.Queue()
    
    @classmethod
    def load_model(
        cls,
        loadModelArgs: LoadModelArgs
        ):
        """Load the models"""

        cls.isLoading = True

        cls.model = AutoPipelineForText2Image.from_pretrained(
            loadModelArgs.model_path,
            torch_dtype = str_to_dtype(loadModelArgs.dtype),
            trust_remote_code = True,
        ).to("mps")

        cls.isLoading = False

    @classmethod 
    def unload_model(
        cls,
    ):
        """Unload model"""

        del cls.model
        del cls.communicationQueue
        del cls.requestQueue
        gc.collect()
        torch.mps.empty_cache()
        cls.model = None
        cls.communicationQueue = {}
        cls.requestQueue = {}

    @classmethod
    async def _service(
        cls,
    ):
        """Use to process requests"""
        if cls.model is None:
            raise BrokenPipeError("Model didn't load.")

        loop = asyncio.get_running_loop()

        while True:

            requestID, requestArg = await cls.requestQueue.get()
            try:
                newImage = await asyncio.to_thread(cls.model, callback_on_step_end= partial(_callback, loop, cls.communicationQueue[requestID], requestArg.num_inference_steps) ,**requestArg.model_dump(exclude=["output_path"]))
                currentTime = datetime.now().strftime('%Y%m%d_%H%M%S')
                newImage.images[0].save(os.path.join(requestArg.output_path, f"{currentTime}.png"))
                loop.call_soon_threadsafe(cls.communicationQueue[requestID].put_nowait, Progress(result= str(os.path.join(requestArg.output_path, f"{currentTime}.png")), percentage= 100.0, status= "success").dict())
                loop.call_soon_threadsafe(cls.communicationQueue[requestID].put_nowait, None)
            except Exception as error:
                loop.call_soon_threadsafe(cls.communicationQueue[requestID].put_nowait, Progress(result= str(error), percentage= 100.0, status= "failed").dict())
                loop.call_soon_threadsafe(cls.communicationQueue[requestID].put_nowait, None)
            finally:
                cls.requestQueue.task_done()

    @classmethod
    async def interface(cls, args: RequestArgs):

        if cls.model is None:
            raise BrokenPipeError("Model didn't load.") 
        
        id = uuid4().hex
        tracker = asyncio.Queue()
        cls.communicationQueue[id] = tracker
        await cls.requestQueue.put((id, args))
        async def get_progress():
            """Use an async Generater to return progress"""
            try:
                while True:
                    progress = await tracker.get()
                    if not progress:
                        break
                    yield f"data: {json.dumps(progress)}\n\n"
                    await asyncio.sleep(0)
            finally:
                cls.communicationQueue.pop(id, None)
            
        return get_progress()