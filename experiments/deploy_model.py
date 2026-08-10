"""
deploy_model.py — Script de Despliegue Final
============================================

Entrena el modelo final con TODOS los datos (100%) usando:
- Mejores hiperparámetros encontrados en tune_experiment
- Mejor threshold encontrado en tune_threshold

Este modelo es el que se usaría en producción.

Uso:
    python experiments/deploy_model.py --params exp009  # Usa params de exp009
    python experiments/deploy_model.py --manual        # Define params manualmente
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Agregar raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from src.data import load_experiment
from src.features import select_features, preprocess_data
from src.models import train_model, evaluate_model, get_model_name


def load_full_dataset(config: dict, df: pd.DataFrame):
    """
    Carga TODOS los datos (sin split) para entrenamiento final.
    
    Args:
        config: Configuración del experimento
        df: DataFrame completo
    
    Returns:
        X, y — Datos completos para entrenamiento
    """
    print("\n" + "=" * 70)
    print("  CARGANDO DATOS COMPLETOS (100%)")
    print("=" * 70)
    
    # Seleccionar features
    features_file = config["data"]["features_file"]
    features = select_features(df, features_file)
    
    target = config["data"]["target"]
    id_column = config["data"]["id_column"]
    
    # Separar X e y (sin split)
    X = df[features].copy()
    y = df[target].copy()
    
    print(f"\n  Dataset completo:")
    print(f"    - Total rows: {len(X):,}")
    print(f"    - Features: {len(features)}")
    print(f"    - Target distribution:")
    print(f"      Class 0 (paid): {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
    print(f"      Class 1 (default): {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
    
    return X, y


def deploy_model(experiment_name: str, output_path: str = None):
    """
    Entrena el modelo final con todos los datos y lo guarda para producción.
    
    Args:
        experiment_name: Nombre del experimento con mejores params (ej: "exp009")
        output_path: Ruta donde guardar el modelo (opcional)
    
    Returns:
        Modelo entrenado listo para producción
    """
    print("\n" + "=" * 70)
    print("  DESPLIEGUE DEL MODELO FINAL")
    print("=" * 70)
    
    # 1. Cargar configuración y datos
    config, df, features = load_experiment(experiment_name)
    
    # 2. Cargar TODOS los datos (sin split)
    X, y = load_full_dataset(config, df)
    
    # 3. Entrenar modelo con TODOS los datos
    print("\n" + "=" * 70)
    print("  ENTRENANDO MODELO FINAL")
    print("=" * 70)
    print("\n  Usando mejores hiperparámetros:")
    for key, value in config["model"]["params"].items():
        print(f"    - {key}: {value}")
    
    pipeline = train_model(config, X, y)
    
    # 4. Evaluar en los mismos datos (solo informativo)
    print("\n" + "=" * 70)
    print("  MÉTRICAS EN DATOS DE ENTRENAMIENTO (informativo)")
    print("=" * 70)
    print("\n  NOTA: Estas métricas son optimistas (entrenamiento)")
    print("  La evaluación real fue en ETAPA 4 (test set)")
    
    metrics = evaluate_model(pipeline, X, y, config, verbose=True)
    
    # 5. Guardar modelo para producción
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "models" / f"deployed_model_{timestamp}.pkl"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo + metadata
    model_artifact = {
        "model": pipeline,
        "config": config,
        "features": features,
        "threshold": config.get("evaluation", {}).get("threshold", None),
        "trained_at": datetime.now().isoformat(),
        "dataset_rows": len(X)
    }
    
    joblib.dump(model_artifact, output_path)
    
    print("\n" + "=" * 70)
    print("  MODELO GUARDADO PARA PRODUCCIÓN")
    print("=" * 70)
    print(f"\n  Ruta: {output_path}")
    print(f"  Threshold: {model_artifact['threshold']}")
    print(f"  Features: {len(features)}")
    print(f"  Trained at: {model_artifact['trained_at']}")
    
    # 6. Registrar en MLflow (opcional)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run(run_name=f"deploy_{experiment_name}"):
        mlflow.log_params(config["model"]["params"])
        mlflow.log_param("threshold", model_artifact["threshold"])
        mlflow.log_param("dataset_rows", len(X))
        mlflow.log_param("deployed", True)
        mlflow.log_metrics(metrics)
        
        # Loggear modelo
        model_name = config["model"]["name"].lower()
        log_kwargs = {
            "sk_model": pipeline,
            "artifact_path": "model",
            "registered_model_name": config["mlflow"].get("registered_model_name", None)
        }
        if "xgboost" in model_name:
            log_kwargs["skops_trusted_types"] = [
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
                "xgboost.sklearn.XGBRegressor"
            ]
        mlflow.sklearn.log_model(**log_kwargs)
    
    print("\n" + "=" * 70)
    print("  DESPLIEGUE COMPLETADO")
    print("=" * 70)
    
    return pipeline


def load_deployed_model(model_path: str):
    """
    Carga un modelo desplegado para hacer predicciones.
    
    Args:
        model_path: Ruta al modelo guardado
    
    Returns:
        Modelo listo para predecir
    """
    artifact = joblib.load(model_path)
    
    print("Modelo cargado:")
    print(f"  - Threshold: {artifact['threshold']}")
    print(f"  - Features: {len(artifact['features'])}")
    print(f"  - Trained at: {artifact['trained_at']}")
    
    return artifact


def predict_new_data(model_artifact: dict, X_new: pd.DataFrame):
    """
    Hace predicciones con el modelo desplegado.
    
    Args:
        model_artifact: Modelo cargado con load_deployed_model()
        X_new: Nuevos datos para predecir
    
    Returns:
        Predicciones (0 o 1)
    """
    pipeline = model_artifact["model"]
    threshold = model_artifact["threshold"]
    features = model_artifact["features"]
    
    # Verificar que tenemos las features correctas
    missing = set(features) - set(X_new.columns)
    if missing:
        raise ValueError(f"Faltan features en los datos: {missing}")
    
    # Seleccionar solo las features del modelo
    X_selected = X_new[features]
    
    # Predecir
    y_proba = pipeline.predict_proba(X_selected)[:, 1]
    
    if threshold is not None:
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = pipeline.predict(X_selected)
    
    return y_pred, y_proba


def main():
    """Comandos del script de despliegue."""
    parser = argparse.ArgumentParser(
        description="Desplegar modelo final entrenado con todos los datos"
    )
    parser.add_argument(
        "--params",
        type=str,
        default="exp009",
        help="Experimento con mejores params (default: exp009)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta donde guardar el modelo (opcional)"
    )
    
    args = parser.parse_args()
    
    deploy_model(args.params, args.output)


if __name__ == "__main__":
    main()
