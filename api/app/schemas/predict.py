from typing import Any, List, Optional

from pydantic import BaseModel
from model.processing.validation import DataInputSchema


# Esquema de una predicción individual
class PredictionItem(BaseModel):
    probabilidad_desempleo: float
    nivel_riesgo: str
    prediccion: str


# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[PredictionItem]]
    umbral_usado: Optional[float]


# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "sexo": "Masculino",
                        "grupo_edad": "18-24",
                        "nivel_educativo": "Media (academica/tecnica)",
                        "etnia": "Ningun grupo etnico",
                        "discapacidad": 0,
                        "jefe_hogar": 0,
                        "mayor_18": 1,
                        "region": "Andina",
                        "zona": "Cabecera",
                        "estrato": 2,
                        "hacinamiento": 1.5,
                        "servicios_basicos_score": 4,
                        "tenencia": "Arrendada",
                        "n_menores_15": 1,
                        "n_mayores_65": 0,
                        "razon_dependencia": 0.5,
                        "inclusion_fin_score": 1,
                        "sin_producto_fin": 0,
                        "transferencias_gov": 0,
                        "recibe_remesas": 0,
                    }
                ]
            }
        }
