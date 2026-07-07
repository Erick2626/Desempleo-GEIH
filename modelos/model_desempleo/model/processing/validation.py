from typing import List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, ValidationError
from model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con nulos en las variables del modelo."""
    validated = input_data.copy()
    return validated.dropna(subset=config.ml_config.features)


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Valida el DataFrame de entrada contra el esquema."""
    relevant = [c for c in config.ml_config.features if c in input_data.columns]
    validated = input_data[relevant].copy()
    validated = drop_na_inputs(input_data=validated)
    errors = None
    try:
        MultipleDataInputs(
            inputs=validated.replace({pd.NA: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()
    return validated, errors


class DataInputSchema(BaseModel):
    # Categoricas
    sexo: Optional[str]
    grupo_edad: Optional[str]
    nivel_educativo: Optional[str]
    etnia: Optional[str]
    region: Optional[str]
    zona: Optional[str]
    tenencia: Optional[str]
    # Numericas
    discapacidad: Optional[float]
    jefe_hogar: Optional[float]
    mayor_18: Optional[float]
    estrato: Optional[float]
    hacinamiento: Optional[float]
    servicios_basicos_score: Optional[float]
    n_menores_15: Optional[float]
    n_mayores_65: Optional[float]
    razon_dependencia: Optional[float]
    inclusion_fin_score: Optional[float]
    sin_producto_fin: Optional[float]
    transferencias_gov: Optional[float]
    recibe_remesas: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
