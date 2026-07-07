import pandas as pd
import pytest


@pytest.fixture
def sample_input_data():
    # Dos perfiles de ejemplo (uno joven urbano, una adulta profesional)
    return pd.DataFrame([
        {
            "sexo": "Masculino", "grupo_edad": "18-24",
            "nivel_educativo": "Media (academica/tecnica)", "etnia": "Ningun grupo etnico",
            "discapacidad": 0, "jefe_hogar": 0, "mayor_18": 1,
            "region": "Andina", "zona": "Cabecera", "estrato": 2,
            "hacinamiento": 1.5, "servicios_basicos_score": 4, "tenencia": "Arrendada",
            "n_menores_15": 1, "n_mayores_65": 0, "razon_dependencia": 0.5,
            "inclusion_fin_score": 1, "sin_producto_fin": 0,
            "transferencias_gov": 0, "recibe_remesas": 0,
        },
        {
            "sexo": "Femenino", "grupo_edad": "45-54",
            "nivel_educativo": "Universitaria", "etnia": "Ningun grupo etnico",
            "discapacidad": 0, "jefe_hogar": 1, "mayor_18": 1,
            "region": "Bogota DC", "zona": "Cabecera", "estrato": 4,
            "hacinamiento": 0.8, "servicios_basicos_score": 5, "tenencia": "Propia",
            "n_menores_15": 0, "n_mayores_65": 0, "razon_dependencia": 0.2,
            "inclusion_fin_score": 3, "sin_producto_fin": 0,
            "transferencias_gov": 0, "recibe_remesas": 0,
        },
    ])
