"""
Sensibilidad de XGBoost a sus hiperparametros, registrada en MLflow.

Complementa la comparacion de algoritmos (run_experiments.py: Arbol vs
XGBoost) con una comparacion de hiperparametros del mismo algoritmo
(XGBoost), siguiendo el mismo enfoque del taller de MLflow: variar un
parametro a la vez frente a la configuracion base (la ya desplegada) y
observar el efecto en las metricas.

Corridas:
  - xgb_base            : configuracion desplegada (eta=0.05, max_depth=8,
                           n_estimators=200) - punto de referencia.
  - xgb_menos_estimadores: n_estimators=50 (menos arboles).
  - xgb_menos_profundidad: max_depth=3 (arboles menos profundos).
  - xgb_mayor_eta        : eta=0.3 (tasa de aprendizaje mas agresiva).

Mismo split 70/30 estratificado y balanceo por submuestreo que
run_experiments.py, para que las 4 corridas sean comparables entre si.

Uso:
    python experiments/run_hyperparam_variations.py --tracking-uri http://<IP-EC2>
"""
import argparse

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from model.config.core import config
from model.processing.data_manager import load_dataset


def build_preprocessor() -> ColumnTransformer:
    cfg = config.ml_config
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cfg.categorical_vars),
        ],
        remainder="passthrough",
    )


def submuestreo(X: pd.DataFrame, y: pd.Series, seed: int):
    idx_d = y[y == 1].index
    idx_o = y[y == 0].index
    rng = np.random.RandomState(seed)
    idx_o_sub = rng.choice(idx_o, size=len(idx_d), replace=False)
    idx = np.concatenate([idx_d, idx_o_sub])
    rng.shuffle(idx)
    return X.loc[idx], y.loc[idx]


def youden_threshold(y_true, proba) -> float:
    fpr, tpr, thr = roc_curve(y_true, proba)
    return float(thr[np.argmax(tpr - fpr)])


def evaluate(pipe, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    thr = youden_threshold(y_test, proba)
    preds = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    return {
        "auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "threshold": thr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_experiment(name: str, X_train, y_train, X_test, y_test, params: dict):
    with mlflow.start_run(run_name=name):
        pipe = SkPipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", XGBClassifier(**params)),
        ])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)

        mlflow.set_tag("algoritmo", "xgboost")
        mlflow.set_tag("balanceo", "submuestreo")
        mlflow.log_params(params)
        mlflow.log_params({"balanceo": "submuestreo", "n_train": len(X_train), "n_test": len(X_test)})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k not in ("tn", "fp", "fn", "tp")})
        mlflow.log_metrics({f"cm_{k}": v for k, v in metrics.items() if k in ("tn", "fp", "fn", "tp")})
        mlflow.sklearn.log_model(pipe, name="model")

        print(f"{name:24s} -> " + " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
        return metrics


def main(tracking_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    cfg = config.ml_config
    data = load_dataset(file_name=config.app_config.training_data_file)
    X = data[cfg.features]
    y = data[cfg.target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    X_bal, y_bal = submuestreo(X_train, y_train, cfg.random_state)

    base = dict(
        objective="binary:logistic", eval_metric="auc",
        eta=cfg.eta, max_depth=cfg.max_depth, n_estimators=cfg.n_estimators,
        subsample=cfg.subsample, random_state=cfg.random_state,
    )

    variaciones = {
        "xgb_base": base,
        "xgb_menos_estimadores": {**base, "n_estimators": 50},
        "xgb_menos_profundidad": {**base, "max_depth": 3},
        "xgb_mayor_eta": {**base, "eta": 0.3},
    }

    resultados = {}
    for nombre, params in variaciones.items():
        resultados[nombre] = run_experiment(nombre, X_bal, y_bal, X_test, y_test, params)

    print("\n=== Resumen comparativo (sensibilidad de hiperparametros) ===")
    print(f"{'corrida':24s} {'AUC':>8s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s}")
    for nombre, m in resultados.items():
        print(f"{nombre:24s} {m['auc']:8.4f} {m['f1']:8.4f} {m['precision']:10.4f} {m['recall']:8.4f}")

    ganador = max(resultados, key=lambda k: resultados[k]["auc"])
    print(f"\nMejor combinacion por AUC: {ganador}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="./mlruns")
    parser.add_argument("--experiment-name", default="desempleo-geih")
    args = parser.parse_args()
    main(args.tracking_uri, args.experiment_name)
