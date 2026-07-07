from model.predict import make_prediction


def test_make_prediction(sample_input_data):
    result = make_prediction(input_data=sample_input_data)

    # Estructura de la respuesta
    assert result["errors"] is None
    assert result["predictions"] is not None
    assert len(result["predictions"]) == 2

    # Cada prediccion tiene los campos esperados
    for pred in result["predictions"]:
        assert 0.0 <= pred["probabilidad_desempleo"] <= 1.0
        assert pred["nivel_riesgo"] in {"Baja", "Media", "Alta"}
        assert pred["prediccion"] in {"Desempleado", "No desempleado"}

    # El perfil joven debe tener mayor probabilidad que la adulta profesional
    p_joven = result["predictions"][0]["probabilidad_desempleo"]
    p_adulta = result["predictions"][1]["probabilidad_desempleo"]
    assert p_joven > p_adulta
