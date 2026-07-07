from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from model.config.core import config

# Preprocesamiento: one-hot a categoricas, passthrough a numericas
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         config.ml_config.categorical_vars),
    ],
    remainder="passthrough",
)

# Pipeline completo (preprocesamiento + XGBoost)
desempleo_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            eta=config.ml_config.eta,
            max_depth=config.ml_config.max_depth,
            n_estimators=config.ml_config.n_estimators,
            subsample=config.ml_config.subsample,
            random_state=config.ml_config.random_state,
        )),
    ]
)
