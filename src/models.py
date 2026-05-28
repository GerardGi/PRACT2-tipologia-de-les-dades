"""
Pipelines de modelat sobre el dataset processat de la P2.

Regressió supervisada sobre `log_price_per_m2`:
- Baseline lineal (OLS)
- Ridge i Lasso amb cerca automàtica d'alpha
- XGBoost amb cerca lleugera d'hiperparàmetres

Clustering no supervisat:
- K-means amb selecció de k via mètode del colze + coeficient de silueta

Convencions:
- Tots els models reben `random_state=42`.
- Split train/test 80/20 estratificat per `province`.
- Feature selection definida explícitament a `NUMERIC_FEATURES` /
  `CATEGORICAL_FEATURES` per a reproductibilitat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score


RANDOM_STATE = 42

# Features triades per al modelat. Justificació:
# - log_area_m2 en lloc d'area_m2 perquè la transformació log redueix l'efecte
#   d'outliers (l'EDA va confirmar la fortíssima asimetria de l'àrea).
# - latitude i longitude entren com a numèriques crues (no log) — el model
#   no lineal (XGBoost) pot capturar el patró geogràfic millor que dist_bcn_km
#   sola; els lineals capturen menys, però dist_bcn_km els ajuda.
# - has_elevator és l'únic boolean del llistat (els de detall tenen 66% NaN i
#   els hem decidit excloure del modelat principal).
NUMERIC_FEATURES = [
    "log_area_m2", "rooms", "bathrooms", "antiguitat_anys", "floor_num",
    "energy_cert_ord", "dist_bcn_km", "n_anuncis_locality",
    "latitude", "longitude",
]
CATEGORICAL_FEATURES = ["province", "property_type", "advertiser_type"]
BOOLEAN_FEATURES = ["has_elevator"]
TARGET = "log_price_per_m2"


def prepare_modeling_dataset(df: pd.DataFrame,
                             extra_filter=None) -> tuple[pd.DataFrame, pd.Series]:
    """Selecciona features i target, aplica filtres estàndard.

    Filtres aplicats:
        - is_price_anomaly == False  (sense outliers de preu)
        - province != "Fora Catalunya"
        - target no nul

    Retorna (X, y).
    """
    mask = (~df["is_price_anomaly"]) & (df["province"] != "Fora Catalunya") & df[TARGET].notna()
    if extra_filter is not None:
        mask = mask & extra_filter(df)
    sub = df.loc[mask].copy()

    feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
    # has_elevator pot tenir dtype bool; el convertim a int per al pipeline
    sub["has_elevator"] = sub["has_elevator"].astype(int)
    return sub[feats].copy(), sub[TARGET].copy()


def build_preprocessor() -> ColumnTransformer:
    """Construeix el ColumnTransformer estàndard del projecte."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                     stratify_col: str = "province",
                     random_state: int = RANDOM_STATE):
    """Split train/test estratificat per `province` (les diferències
    significatives entre províncies justifiquen no fer split aleatori pur —
    algunes podrien quedar sub-representades)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state,
                            stratify=X[stratify_col])


def build_linear_pipeline() -> Pipeline:
    return Pipeline([("prep", build_preprocessor()),
                     ("model", LinearRegression())])


def build_ridge_pipeline(alphas=(0.01, 0.1, 1.0, 10.0, 100.0)) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()),
                     ("model", RidgeCV(alphas=alphas, cv=5,
                                       scoring="neg_mean_squared_error"))])


def build_lasso_pipeline(alphas=(0.0001, 0.001, 0.01, 0.1, 1.0)) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()),
                     ("model", LassoCV(alphas=alphas, cv=5,
                                       random_state=RANDOM_STATE, max_iter=5000))])


def build_xgboost_pipeline(n_estimators: int = 500, max_depth: int = 6,
                           learning_rate: float = 0.05,
                           subsample: float = 0.9,
                           colsample_bytree: float = 0.9,
                           **extra_xgb_kwargs) -> Pipeline:
    """Pipeline XGBoost. `**extra_xgb_kwargs` permet passar reg_alpha,
    reg_lambda, gamma, etc. (útil amb Optuna)."""
    from xgboost import XGBRegressor
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, random_state=RANDOM_STATE,
            tree_method="hist", n_jobs=-1, verbosity=0,
            **extra_xgb_kwargs)),
    ])


def evaluate_model(model, X_test, y_test, name: str = "model") -> dict:
    """Mètriques en escala log i en escala original (€/m²)."""
    y_pred = model.predict(X_test)
    res = {
        "model": name,
        "n_test": len(y_test),
        "R2": float(r2_score(y_test, y_pred)),
        "MAE_log": float(mean_absolute_error(y_test, y_pred)),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        # Recuperació a l'escala original: si log_ppm2 = a, llavors ppm2 = e^a
        "MAE_eur_m2": float(mean_absolute_error(np.exp(y_test), np.exp(y_pred))),
        "RMSE_eur_m2": float(np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred)))),
        "MdAPE_pct": float(np.median(np.abs(np.exp(y_pred) - np.exp(y_test)) / np.exp(y_test)) * 100),
    }
    return res


def cv_evaluate(pipe, X, y, n_splits: int = 5,
                random_state: int = RANDOM_STATE) -> dict:
    """K-fold CV (k=5) sobre el dataset. Retorna mitjana ± std de R² i MAE_log."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2s, maes = [], []
    for fold, (tr, te) in enumerate(kf.split(X)):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        y_pred = pipe.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], y_pred))
        maes.append(mean_absolute_error(y.iloc[te], y_pred))
    return {
        "cv_R2_mean": float(np.mean(r2s)), "cv_R2_std": float(np.std(r2s)),
        "cv_MAE_log_mean": float(np.mean(maes)), "cv_MAE_log_std": float(np.std(maes)),
        "n_splits": n_splits,
    }


def get_feature_names(pipe: Pipeline) -> list[str]:
    """Recupera els noms de features post-preprocessor (incloent one-hot)."""
    prep = pipe.named_steps["prep"]
    names = list(NUMERIC_FEATURES)
    for name, trans, _cols in prep.transformers_:
        if name == "cat" and hasattr(trans, "get_feature_names_out"):
            names += trans.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
            break
    names += list(BOOLEAN_FEATURES)
    return names


# =====================================================================
# CLUSTERING — K-means
# =====================================================================

CLUSTERING_NUMERIC = [
    "log_price_per_m2", "log_area_m2", "rooms", "bathrooms",
    "antiguitat_anys", "energy_cert_ord", "dist_bcn_km",
]


def prepare_clustering_dataset(df: pd.DataFrame) -> pd.DataFrame:
    mask = ((~df["is_price_anomaly"]) & (df["province"] != "Fora Catalunya")
            & df["log_price_per_m2"].notna())
    sub = df.loc[mask, CLUSTERING_NUMERIC + ["province", "property_type", "locality"]].copy()
    return sub.dropna(subset=CLUSTERING_NUMERIC)


def kmeans_scan(X_scaled: np.ndarray, ks: range = range(2, 11),
                random_state: int = RANDOM_STATE,
                silhouette_sample: int = 5000) -> pd.DataFrame:
    """Calcula inèrcia i silueta per a cada k. Per estalviar temps, la silueta
    es calcula sobre una submostra (n=5.000) per a k > 2."""
    rng = np.random.default_rng(random_state)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        # Silueta sobre submostra (millor escalabilitat amb N gran)
        if len(X_scaled) > silhouette_sample:
            idx = rng.choice(len(X_scaled), silhouette_sample, replace=False)
            sil = silhouette_score(X_scaled[idx], km.labels_[idx])
        else:
            sil = silhouette_score(X_scaled, km.labels_)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def kmeans_fit(X_scaled: np.ndarray, k: int,
               random_state: int = RANDOM_STATE) -> tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = km.fit_predict(X_scaled)
    return km, labels


def cluster_profile(df_with_labels: pd.DataFrame, label_col: str = "cluster",
                    numeric_cols=None, categorical_cols=None) -> pd.DataFrame:
    """Perfil descriptiu per cluster: mediana de numèriques + mode de categòriques."""
    numeric_cols = numeric_cols or CLUSTERING_NUMERIC
    categorical_cols = categorical_cols or ["province", "property_type"]
    profile = df_with_labels.groupby(label_col)[numeric_cols].median().round(3)
    profile["n_anuncis"] = df_with_labels.groupby(label_col).size()
    for c in categorical_cols:
        profile[f"top_{c}"] = (df_with_labels.groupby(label_col)[c]
                               .agg(lambda x: x.value_counts().idxmax()))
    cols = ["n_anuncis"] + numeric_cols + [f"top_{c}" for c in categorical_cols]
    return profile[cols]


def tune_xgboost_optuna(X_train, y_train, n_trials: int = 50,
                       cv_splits: int = 5,
                       random_state: int = RANDOM_STATE,
                       verbose: bool = False) -> dict:
    """Cerca bayesiana d'hiperparàmetres d'XGBoost amb Optuna.

    L'espai de cerca cobreix els paràmetres més influents (n_estimators,
    max_depth, learning_rate, subsample, colsample_bytree, reg_alpha,
    reg_lambda) en escales adequades (log per learning_rate i regularitzadors).

    Retorna {"best_params", "best_score", "trials_df"}. Optimitza R² mitjà
    via CV k=5 sobre el conjunt d'entrenament.
    """
    import optuna
    from sklearn.model_selection import KFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        pipe = build_xgboost_pipeline(**params)
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X_train, y_train, cv=kf,
                                 scoring="r2", n_jobs=1)
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe()[["number", "value", "params_n_estimators",
                                          "params_max_depth", "params_learning_rate"]]
    return {
        "best_params": study.best_params,
        "best_score": float(study.best_value),
        "trials_df": trials_df,
        "n_trials": n_trials,
    }


def spatial_cv_eval(pipe, X: pd.DataFrame, y: pd.Series,
                    group_col: str = "province",
                    random_state: int = RANDOM_STATE) -> dict:
    """Avaluació amb GroupKFold per `group_col` — més conservadora davant
    autocorrelació espacial.

    Cada fold deixa fora un grup sencer; per a `province` amb 4 valors,
    es fan exactament 4 folds (leave-one-province-out)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score, mean_absolute_error

    groups = X[group_col].values
    n_splits = X[group_col].nunique()
    gkf = GroupKFold(n_splits=n_splits)
    r2s, maes, fold_names = [], [], []
    for tr, te in gkf.split(X, y, groups=groups):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        y_pred = pipe.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], y_pred))
        maes.append(mean_absolute_error(y.iloc[te], y_pred))
        fold_names.append(X.iloc[te][group_col].iloc[0])
    return {
        "n_splits": n_splits,
        "groups": fold_names,
        "r2_per_fold": [float(s) for s in r2s],
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "mae_log_mean": float(np.mean(maes)), "mae_log_std": float(np.std(maes)),
    }


# Compat amb stubs antics
def fit_baseline_regression(X, y):
    pipe = build_linear_pipeline()
    pipe.fit(X, y)
    return pipe


def fit_xgboost_regression(X, y, **kwargs):
    pipe = build_xgboost_pipeline(**kwargs)
    pipe.fit(X, y)
    return pipe


def fit_kmeans_segments(X_scaled, k, random_state=RANDOM_STATE):
    return kmeans_fit(X_scaled, k, random_state)
