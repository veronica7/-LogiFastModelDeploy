from pydantic import BaseModel, Field
from typing import Dict

class PredictionRequest(BaseModel):
    pickuplocation: str = Field(..., description="Indirizzo di ritiro del pacco.")
    deliverylocation: str = Field(..., description="Indirizzo di consegna del pacco.")
   ## pickupdatetime: str = Field(..., description="Data e ora di ritiro del pacco (formato ISO 8601).")
    weight: float = Field(..., description="Peso del pacco in chilogrammi.")
    servicetype: str = Field(..., description="Tipo di servizio richiesto (es. standard, express, same-day).")
    

class PredictionOutput(BaseModel):
    estimated_delivery_time:  float = Field(..., description='Predicted estimated delivery time.')
    