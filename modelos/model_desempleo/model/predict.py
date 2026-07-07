import typing as t
import pandas as pd
from model import __version__ as _version
from model.config.core import config
from model.processing.data_manager import load_pipeline
from model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_desempleo_pipe = load_pipeline(file_name=pipeline_file_name)


def _nivel_riesgo(p: float) -> str:
    thr = config.ml_config.decision_threshold
    if p < 0.35:
        return "Baja"
    elif p < thr:
        return "Media"
    return "Alta"


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
    """Genera prediccion de desempleo para uno o varios perfiles."""
    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if errors:
        return results

    thr = config.ml_config.decision_threshold
    proba = _desempleo_pipe.predict_proba(
        validated_data[config.ml_config.features]
    )[:, 1]

    results["predictions"] = [
        {
            "probabilidad_desempleo": round(float(p), 4),
            "nivel_riesgo": _nivel_riesgo(float(p)),
            "prediccion": "Desempleado" if p > thr else "No desempleado",
        }
        for p in proba
    ]
    results["umbral_usado"] = round(thr, 3)
    return results
