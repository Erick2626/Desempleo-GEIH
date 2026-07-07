import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("http://localhost:8001/api/v1/health")

    assert response.status_code == 200
    result = response.json()
    assert result["name"]
    assert result["api_version"]
    assert result["model_version"]


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    # Given
    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }

    # When
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["predictions"]
    assert prediction_data["errors"] is None
    assert len(prediction_data["predictions"]) == 2

    for pred in prediction_data["predictions"]:
        assert 0.0 <= pred["probabilidad_desempleo"] <= 1.0
        assert pred["nivel_riesgo"] in {"Baja", "Media", "Alta"}
        assert pred["prediccion"] in {"Desempleado", "No desempleado"}

    # El perfil joven (0) debe tener mayor probabilidad que la adulta profesional (1)
    p_joven = prediction_data["predictions"][0]["probabilidad_desempleo"]
    p_adulta = prediction_data["predictions"][1]["probabilidad_desempleo"]
    assert p_joven > p_adulta
