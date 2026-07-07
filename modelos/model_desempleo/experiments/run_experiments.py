"""
Comparacion de modelos supervisados para la prediccion de desempleo,
registrada en MLflow.

Entrena y compara un Arbol de decision y un XGBoost, ambos sobre el mismo
split 70/30 estratificado y con balanceo por submuestreo 50/50 en train.

Cada corrida registra en MLflow: parametros, metricas (AUC-ROC, F1, precision,
recall sobre la clase 1, con umbral de Youden) y el pipeline entrenado
(preprocesamiento + modelo) como artefacto.

Uso:
    python experiments/run_experiments.py --tracking-uri http://<IP-EC2>
    python experiments/run_experiments.py                     # usa ./mlruns local

Requiere model/datasets/df_modelo.csv presente (mismo dataset del train_pipeline).
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
from sklearn.tree import DecisionTreeClassifier
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
    """Balanceo por submuestreo: todas las D + misma cantidad de O."""
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


def run_experiment(name: str, algoritmo: str, pipe, X_train, y_train, X_test, y_test, params: dict):
    with mlflow.start_run(run_name=name):
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)

        mlflow.set_tag("algoritmo", algoritmo)
        mlflow.set_tag("balanceo", "submuestreo")
        mlflow.log_params(params)
        mlflow.log_params({"balanceo": "submuestreo", "n_train": len(X_train), "n_test": len(X_test)})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k not in ("tn", "fp", "fn", "tp")})
        mlflow.log_metrics({f"cm_{k}": v for k, v in metrics.items() if k in ("tn", "fp", "fn", "tp")})
        mlflow.sklearn.log_model(pipe, name="model")

        print(f"{name:22s} -> " + " | ".join(
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
    # Balanceo por submuestreo en train (ambos modelos entrenan sobre el mismo set)
    X_bal, y_bal = submuestreo(X_train, y_train, cfg.random_state)

    resultados = {}

    # Modelo 1: Arbol de decision
    params_arbol = {"max_depth": 8, "min_samples_leaf": 50, "criterion": "gini"}
    pipe_arbol = SkPipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DecisionTreeClassifier(random_state=cfg.random_state, **params_arbol)),
    ])
    resultados["arbol_decision"] = run_experiment(
        "arbol_decision", "arbol_decision", pipe_arbol, X_bal, y_bal, X_test, y_test, params_arbol
    )

    # Modelo 2: XGBoost (mejor combinacion del grid search) - modelo desplegado
    params_xgb = dict(
        objective="binary:logistic", eval_metric="auc",
        eta=cfg.eta, max_depth=cfg.max_depth, n_estimators=cfg.n_estimators,
        subsample=cfg.subsample, random_state=cfg.random_state,
    )
    pipe_xgb = SkPipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(**params_xgb)),
    ])
    resultados["xgboost"] = run_experiment(
        "xgboost_submuestreo", "xgboost", pipe_xgb, X_bal, y_bal, X_test, y_test, params_xgb
    )

    print("\n=== Resumen comparativo ===")
    print(f"{'modelo':20s} {'AUC':>8s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s}")
    for nombre, m in resultados.items():
        print(f"{nombre:20s} {m['auc']:8.4f} {m['f1']:8.4f} {m['precision']:10.4f} {m['recall']:8.4f}")

    ganador = max(resultados, key=lambda k: resultados[k]["auc"])
    print(f"\nModelo con mejor AUC: {ganador}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="./mlruns", help="URI del tracking server de MLflow (ej. http://IP). Por defecto usa ./mlruns local.")
    parser.add_argument("--experiment-name", default="desempleo-geih")
    args = parser.parse_args()
    main(args.tracking_uri, args.experiment_name)
