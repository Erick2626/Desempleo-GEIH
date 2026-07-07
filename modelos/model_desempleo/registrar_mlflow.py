"""
Registra los 3 experimentos de modelos en el MLflow de la EC2.
Uso:  python registrar_mlflow.py
Ajusta la IP si el lab reinicio la instancia.
"""

import mlflow

mlflow.set_tracking_uri("http://100.56.244.14:5000")
mlflow.set_experiment("desempleo-geih")

experimentos = [
    (
        "arbol_decision",
        {"algoritmo": "tree", "balanceo": "submuestreo", "cp": 0.001},
        {"auc": 0.664, "sensibilidad": 0.621, "especificidad": 0.646, "f1": 0.282},
    ),
    (
        "xgboost_submuestreo",
        {
            "algoritmo": "xgboost",
            "balanceo": "submuestreo",
            "eta": 0.05,
            "max_depth": 8,
            "n_estimators": 200,
            "threshold": 0.485,
        },
        {"auc": 0.7132, "sensibilidad": 0.666, "especificidad": 0.647, "f1": 0.300},
    ),
    (
        "xgboost_scale_pos_weight",
        {
            "algoritmo": "xgboost",
            "balanceo": "scale_pos_weight",
            "eta": 0.05,
            "max_depth": 8,
            "scale_pos_weight": 7.85,
        },
        {"auc": 0.713, "sensibilidad": 0.657, "especificidad": 0.658, "f1": 0.303},
    ),
]

for nombre, params, metrics in experimentos:
    with mlflow.start_run(run_name=nombre):
        mlflow.set_tag("modelo", nombre)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    print("Registrado:", nombre)

print("LISTO - 3 experimentos registrados en MLflow")
