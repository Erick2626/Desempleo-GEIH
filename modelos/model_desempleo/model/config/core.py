from pathlib import Path
from typing import List
from pydantic import BaseModel
from strictyaml import load
import model

# Directorios del paquete (PACKAGE_ROOT = carpeta model/, tanto en modo
# editable como una vez instalado el wheel en site-packages)
PACKAGE_ROOT = Path(model.__file__).resolve().parent
ROOT = PACKAGE_ROOT
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"


class AppConfig(BaseModel):
    package_name: str
    training_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    target: str
    decision_threshold: float
    features: List[str]
    categorical_vars: List[str]
    numerical_vars: List[str]
    test_size: float
    random_state: int
    eta: float
    max_depth: int
    n_estimators: int
    subsample: float


class Config(BaseModel):
    app_config: AppConfig
    # Nota: no se llama "model_config" a propósito - ese nombre está
    # reservado internamente por pydantic 2.x (ConfigDict).
    ml_config: ModelConfig


def fetch_config_from_yaml():
    if CONFIG_FILE_PATH.is_file():
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            return load(f.read())
    raise OSError(f"No se encontro config.yml en {CONFIG_FILE_PATH}")


def create_and_validate_config():
    parsed = fetch_config_from_yaml()
    d = parsed.data
    return Config(
        app_config=AppConfig(
            package_name=d["package_name"],
            training_data_file=d["training_data_file"],
            pipeline_save_file=d["pipeline_save_file"],
        ),
        ml_config=ModelConfig(
            target=d["target"],
            decision_threshold=float(d["decision_threshold"]),
            features=d["features"],
            categorical_vars=d["categorical_vars"],
            numerical_vars=d["numerical_vars"],
            test_size=float(d["test_size"]),
            random_state=int(d["random_state"]),
            eta=float(d["eta"]),
            max_depth=int(d["max_depth"]),
            n_estimators=int(d["n_estimators"]),
            subsample=float(d["subsample"]),
        ),
    )


config = create_and_validate_config()
