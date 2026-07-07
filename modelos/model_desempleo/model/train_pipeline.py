"""
Entrena el pipeline de desempleo y lo guarda versionado.
Fuerza de trabajo activa, 20 predictores, split 70/30 estratificado,
balanceo por submuestreo 50/50 en train, XGBoost, umbral de Youden.

El XGBoost gano frente al arbol de decision en la comparacion de modelos
(ver experiments/run_experiments.py). El submuestreo produce un umbral de
decision ~0.49, consistente con el tablero.

Uso:  python -m model.train_pipeline
Requiere:  model/datasets/df_modelo.csv
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from model.config.core import config, CONFIG_FILE_PATH
from model.pipeline import desempleo_pipe
from model.processing.data_manager import load_dataset, save_pipeline


def _submuestreo(X: pd.DataFrame, y: pd.Series, seed: int):
    """Balanceo por submuestreo: todas las D + misma cantidad de O."""
    idx_d = y[y == 1].index
    idx_o = y[y == 0].index
    rng = np.random.RandomState(seed)
    idx_o_sub = rng.choice(idx_o, size=len(idx_d), replace=False)
    idx = np.concatenate([idx_d, idx_o_sub])
    rng.shuffle(idx)
    return X.loc[idx], y.loc[idx]


def run_training() -> None:
    cfg = config.ml_config

    # 1. Datos (ya filtrados a fuerza de trabajo activa al exportar)
    data = load_dataset(file_name=config.app_config.training_data_file)
    X = data[cfg.features]
    y = data[cfg.target].astype(int)

    # 2. Split 70/30 estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    # 3. Balanceo por submuestreo en train
    X_bal, y_bal = _submuestreo(X_train, y_train, cfg.random_state)

    # 4. Entrenar pipeline
    desempleo_pipe.fit(X_bal, y_bal)

    # 5. Umbral optimo por Youden en test
    proba_test = desempleo_pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, proba_test)
    youden = tpr - fpr
    thr_opt = float(thr[np.argmax(youden)])
    auc = roc_auc_score(y_test, proba_test)
    print(f"AUC test: {auc:.4f} | umbral Youden: {thr_opt:.3f}")

    # 6. Persistir umbral en config.yml (para predict/nivel de riesgo)
    _actualizar_umbral(thr_opt)

    # 7. Guardar pipeline versionado
    save_pipeline(pipeline_to_persist=desempleo_pipe)
    print("Pipeline guardado.")


def _actualizar_umbral(thr_opt: float) -> None:
    txt = CONFIG_FILE_PATH.read_text(encoding="utf-8")
    nuevas = []
    for line in txt.splitlines():
        if line.strip().startswith("decision_threshold:"):
            nuevas.append(f"decision_threshold: {round(thr_opt, 4)}")
        else:
            nuevas.append(line)
    CONFIG_FILE_PATH.write_text("\n".join(nuevas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_training()
