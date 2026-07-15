from utils.config import read_config
from schemas.inference.text2text import LoadModelArgs

import asyncio
from argparse import Namespace
import requests
import multiprocessing
import threading
import traceback

from mlx_lm import server

class MlxTextGeneration:
    
    def __init__(self) -> None:
        self.service = None
        self.isLoading = False
        self.host = read_config("MLX", "host")
        self.port = read_config("MLX", "port")
        self.modelTag = ""

        if self.host is None:
            raise ValueError("MLX host is not configured in config.ini")

        if self.port is None:
            raise ValueError("MLX port is not configured in config.ini")

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


    async def load_model(
        self,
        loadModelArgs: LoadModelArgs,
    ):

        # Startup servelet
        errorQueue = multiprocessing.Queue()
        self.isLoading = True
        self.service = multiprocessing.Process(target= self._run_server, kwargs={"host": self.host, "port": int(self.port), "modelPath": loadModelArgs.modelPath, "errorQueue": errorQueue })
        self.service.start()
        
        while True:
            # Check Server startup finish or not
            # Check if any error return
            if not errorQueue.empty():
                self.service.terminate()
                self.service = None
                self.isLoading = False
                raise RuntimeError(f"Server startup error: {errorQueue.get()}") 

            try:
                req = requests.get(f"http://{self.host}:{self.port}/v1/models", timeout= 1)
                if req.status_code == 200:
                    self.isLoading = False
                    self.modelTag = loadModelArgs.modelPath
                    break
                raise Exception("Timeout")
            except:
                await asyncio.sleep(1)
                    
    def unload_model(self):
        
        if self.service is not None:
            self.service.terminate()
            self.service = None
            self.modelTag = ""