from pydantic import BaseModel, Field, field_validator
from typing import Dict

VALID_CITIES = {"Ancona","Bari","Bologna","Cagliari","Catania","Firenze","Genova",
                 "Lecce","Milano","Napoli","Palermo","Perugia","Pescara",
                 "Reggio Calabria","Roma","Salerno","Sassari","Torino","Trapani","Verona"}
VALID_SERVICE_TYPES = {"Express", "Premium"}

class PredictionRequest(BaseModel):
    pickuplocation: str = Field(..., description="Indirizzo di ritiro del pacco.")
    deliverylocation: str = Field(..., description="Indirizzo di consegna del pacco.")
    weight: float = Field(..., description="Peso del pacco in chilogrammi.")
    servicetype: str = Field(..., description="Tipo di servizio richiesto (es. standard, express, same-day).")
    
    @field_validator("pickuplocation", "deliverylocation")
    def check_city(cls, v):
        if v not in VALID_CITIES:
            raise ValueError(f"Città non riconosciuta: {v}")
        return v

    @field_validator("servicetype")
    def check_service(cls, v):
        if v not in VALID_SERVICE_TYPES:
            raise ValueError(f"Tipo di servizio non valido: {v}")
        return v

class PredictionOutput(BaseModel):
    estimated_delivery_time:  float = Field(..., description='Predicted estimated delivery time.')
    unit: str = "minutes"
    reliability_score: float = Field(..., description="Intervallo di confidenza basato sui residui del modello.")
    confidence_interval: tuple[float, float]
    