from modules.inference.text2text.mlx import MlxTextGeneration
from schemas.inference.text2text import LoadModelArgs, InferenceArgs


import asyncio

from openai import AsyncOpenAI

class Text2TextInference:

    engine= MlxTextGeneration()
    lock = asyncio.Lock() # Avoid load model and unload model at same time

    @classmethod
    async def load_model(cls, loadModelArgs:LoadModelArgs):
        """load model"""

        if cls.engine.isLoading or cls.engine.service is not None:
            return 400, "Request Error", "Already load model or loading "
        
        async with cls.lock:

            try:
                await cls.engine.load_model(loadModelArgs= loadModelArgs)
            except Exception as error:
                return 500, "Model Load Error", str(error)  
    
            return 200, "Success", "Success"

    @classmethod
    async def unload_model(cls):
        """unload model"""

        if cls.engine.isLoading or cls.engine.service is None:
            return 400, "Request Error", "Didn't load model or loading "
        
        async with cls.lock:

            try:
                cls.engine.unload_model()
            except Exception as error:
                return 500, "Unload Model Error", str(error)
        
            return 200, "Success", "Success"
        
    @classmethod
    async def debug_send_request(
            cls,
            params: InferenceArgs
        ):

        if cls.engine.isLoading or cls.engine.service is None:
            return 400, "Request Error", "Didn't load model or loading "
        
        client = AsyncOpenAI(
            base_url= f"http://{cls.engine.host}:{cls.engine.port}/v1",
            api_key= "None",
        )

        generator = await client.chat.completions.create(
            model= cls.engine.modelTag,
            messages=[
                {"role": "system", "content": params.systemPrompt},
                {"role": "user", "content": params.prompt}
            ],
            temperature= params.temperature,
            top_p= params.topP,
            stream= True,
            max_tokens= params.maxToken
        )

        async def streamer(generator):
            async for chunk in generator:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        return 200, streamer(generator), None 