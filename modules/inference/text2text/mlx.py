from utils.config import read_config

import asyncio
from argparse import Namespace
import requests
import multiprocessing
import threading
import traceback

from mlx_lm import server

class MlxTextGeneration:
    
    service = None

    @staticmethod
    def _run_server(
        modelPath: str,
        host: str,
        port: int,
        errorQueue: multiprocessing.Queue,
    ):
        """wrapper for catch except"""

        def _thread_excepthook(args: threading.ExceptHookArgs):
            # Becaue mlx_lm.server threw except by thread.
            # So use a excepthook to catch except 
            errorMessage = "".join(
                traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
            )
            errorQueue.put(errorMessage)
        # hook
        threading.excepthook = _thread_excepthook

        # Create a ModelProvider to configure parameters
        try:
            provider = server.ModelProvider(
                cli_args= Namespace(
                    model= modelPath,
                    host= host,
                    port= port,
                    # Below use default
                    adapter_path= None,
                    allowed_origins= "*",
                    draft_model= None,
                    num_draft_tokens= 3,
                    trust_remote_code= True,
                    log_level= "INFO",
                    chat_template= "",
                    use_default_chat_template= False,
                    temp= 0.0,
                    top_p= 1.0,
                    top_k= 0,
                    min_p= 0.0,
                    max_tokens= 512,
                    chat_template_args= {},
                    decode_concurrency= 32,
                    prompt_concurrency= 8,
                    prefill_step_size= 2048,
                    prompt_cache_size= 10,
                    prompt_cache_bytes= None,
                    pipeline= False,
                )
            )
            # Startup server
            server.run(
                host= host,
                port= port,
                model_provider= provider
            )
        except Exception as error:
            errorQueue.put(str(error))


    @classmethod
    async def load_model(
        cls,
        modelPath: str
    ):

        host = read_config("MLX", "host") 
        port = read_config("MLX", "port")

        if host is None:
            raise ValueError("MLX host is not configured in config.ini")

        if port is None:
            raise ValueError("MLX port is not configured in config.ini")
        

        # Startup servelet
        errorQueue = multiprocessing.Queue()
        cls.service = multiprocessing.Process(target= cls._run_server, kwargs={"host": host, "port": int(port), "modelPath": modelPath, "errorQueue": errorQueue })
        cls.service.start()
        
        while True:
            # Check Server startup finish or not
            # Check if any error return
            if not errorQueue.empty():
                cls.service.terminate()
                cls.service = None
                raise RuntimeError(f"Server startup error: {errorQueue.get()}") 

            try:
                req = requests.get(f"http://{host}:{port}/v1/models", timeout= 1)
                if req.status_code == 200:
                    break
                raise Exception("Timeout")
            except:
                await asyncio.sleep(1)
                    
    @classmethod
    def unload_model(cls):
        
        if cls.service is not None:
            cls.service.terminate()
            cls.service = None

        
if __name__ == "__main__":
    """debug"""

    going = True
    try:
        asyncio.run(MlxTextGeneration.load_model("/Users/alpha/model/Qwen3"))
    except Exception as error:
        going = False
        print(error)

    if going:
        input("Next?")
        MlxTextGeneration.unload_model()
        input("Next?")