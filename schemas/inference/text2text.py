from pydantic import BaseModel, Field

class LoadModelArgs(BaseModel):

    modelPath: str

class InferenceArgs(BaseModel):
    prompt: str
    maxToken: int 
    systemPrompt : str = Field(default= "")
    temperature: float = Field(default= 1.0)
    topP: float = Field(default= 0.0)

class ServiceInformation(BaseModel):
    """For Get Service Informations. Show host port and modelTag."""
    host: str| None
    port: str| None
    modelTag: str