from typing import Literal

from pydantic import BaseModel

class ConvertArgForMLX(BaseModel):
    modelPath: str
    outputPath: str
    precision: Literal["bfloat16", "int8", "int4", "mxfp4", "mxfp8", "nvfp4"]
    dtype: str = "bfloat16"