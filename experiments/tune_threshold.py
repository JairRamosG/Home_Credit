"""
tune_threshold.py — Búsqueda del mejor umbral de clasificación
==============================================================

Entrena el modelo UNA vez y prueba múltiples umbrales en el validation set
para encontrar el que mejor se adapte al objetivo de negocio.

IMPORTANTE: Este script debe usar el VALIDATION SET, no el test set.
El test set se reserva para la evaluación final.

Uso:
    python experiments/tune_threshold.py exp008
    python experiments/tune_threshold.py exp008 --verbose
"""

import sys
import argparse
from pathlib import Path

# Agregar raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from src.data import load_experiment
from src.features import prepare_data
from src.models import train_model, get_model_name


def calculate_metrics(y_true, y_pred, y_prob):
    """Calcula todas las métricas de evaluación."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }


def tune_threshold(experiment_name: str, verbose: bool = True):
    """
    Búsqueda del mejor umbral de clasificación.
    
    IMPORTANTE: Usa el validation set para buscar el mejor umbral.
    El test set se reserva para la evaluación final.
    
    Args:
        experiment_name: Nombre del experimento YAML (ej: "exp008")
        verbose: Si True, muestra output detallado
    
    Returns:
        DataFrame con resultados de todos los umbrales
    """
    print("\n" + "=" * 70)
    print(f"  THRESHOLD TUNING: {experiment_name.upper()}")
    print("=" * 70)
    
    # 1. Cargar configuración y datos
    config, df, features = load_experiment(experiment_name)
    
    # 2. Preparar datos (con 3-way split si validation_size está definido)
    result = prepare_data(config, df, features)
    
    # 3. Verificar que tenemos validation set
    if len(result) == 6:
        X_train, X_val, X_test, y_train, y_val, y_test = result
        print("\n✓ Usando VALIDATION SET para threshold tuning")
        X_eval = X_val
        y_eval = y_val
        eval_set_name = "validation"
    else:
        print("\n⚠️  WARNING: No hay validation set definido en el config")
        print("   Usando TEST SET (no recomendado para producción)")
        X_train, X_test, y_train, y_test = result
        X_eval = X_test
        y_eval = y_test
        eval_set_name = "test"
    
    # 4. Entrenar modelo UNA vez
    pipeline = train_model(config, X_train, y_train)
    
    # 5. Obtener probabilidades en el set de evaluación
    print(f"\nGenerando probabilidades en {eval_set_name} set...")
    y_prob = pipeline.predict_proba(X_eval)[:, 1]
    
    # 6. Obtener grid de umbrales del config (OBLIGATORIO en el YAML)
    threshold_config = config["evaluation"]["thresholds"]
    threshold_min = threshold_config["min"]
    threshold_max = threshold_config["max"]
    threshold_step = threshold_config["step"]
    
    thresholds = np.arange(threshold_min, threshold_max + threshold_step/2, threshold_step)
    
    print(f"Probando {len(thresholds)} umbrales: {list(thresholds)}")
    print("-" * 70)
    
    # 7. Evaluar cada umbral
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = calculate_metrics(y_eval, y_pred, y_prob)
        metrics['threshold'] = threshold
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_eval, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        results.append(metrics)
        
        if verbose:
            print(f"Threshold: {threshold:.2f} | "
                  f"Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"Accuracy: {metrics['accuracy']:.4f}")
    
    # 8. Crear DataFrame con resultados
    df_results = pd.DataFrame(results)
    
    # 9. Encontrar mejor umbral por cada métrica
    print("\n" + "=" * 70)
    print("MEJORES UMBRALES POR MÉTRICA")
    print("=" * 70)
    
    best_roc_auc = df_results.loc[df_results['roc_auc'].idxmax()]
    best_f1 = df_results.loc[df_results['f1'].idxmax()]
    best_recall = df_results.loc[df_results['recall'].idxmax()]
    best_precision = df_results.loc[df_results['precision'].idxmax()]
    
    print(f"\nMejor para ROC AUC:    threshold={best_roc_auc['threshold']:.2f} "
          f"(ROC AUC={best_roc_auc['roc_auc']:.4f})")
    print(f"Mejor para F1:         threshold={best_f1['threshold']:.2f} "
          f"(F1={best_f1['f1']:.4f})")
    print(f"Mejor para Recall:     threshold={best_recall['threshold']:.2f} "
          f"(Recall={best_recall['recall']:.4f})")
    print(f"Mejor para Precision:  threshold={best_precision['threshold']:.2f} "
          f"(Precision={best_precision['precision']:.4f})")
    
    # 10. Guardar resultados en CSV
    results_path = PROJECT_ROOT / "experiments" / "results" / f"{experiment_name}_thresholds.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_path, index=False)
    print(f"\nResultados guardados en: {results_path}")
    
    # 11. Registrar en MLflow
    log_to_mlflow(config, df_results, pipeline, eval_set_name)
    
    print("\n" + "=" * 70)
    print(f"  THRESHOLD TUNING COMPLETADO")
    print("=" * 70)
    
    return df_results


def log_to_mlflow(config: dict, df_results: pd.DataFrame, pipeline, eval_set_name: str):
    """Registra los resultados del threshold tuning en MLflow."""
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)
    
    model_name = get_model_name(config)
    run_name = f"{config['experiment']['name']}_threshold_tuning"
    
    with mlflow.start_run(run_name=run_name):
        # 1. Loggear parámetros del modelo
        mlflow.log_params(config["model"]["params"])
        
        # 2. Loggear configuración de datos
        mlflow.log_params({
            "dataset": config["data"]["dataset"],
            "features_file": config["data"]["features_file"],
            "test_size": config["data"]["test_size"],
            "random_state": config["data"]["random_state"],
            "validation_size": config["data"].get("validation_size", "N/A")
        })
        
        # 3. Loggear configuración de thresholds
        threshold_config = config["evaluation"]["thresholds"]
        mlflow.log_params({
            "threshold_min": threshold_config["min"],
            "threshold_max": threshold_config["max"],
            "threshold_step": threshold_config["step"],
            "eval_set": eval_set_name
        })
        
        # 4. Loggear mejor umbral por métrica
        best_roc_auc = df_results.loc[df_results['roc_auc'].idxmax()]
        best_f1 = df_results.loc[df_results['f1'].idxmax()]
        best_recall = df_results.loc[df_results['recall'].idxmax()]
        
        mlflow.log_params({
            "best_threshold_roc_auc": float(best_roc_auc['threshold']),
            "best_threshold_f1": float(best_f1['threshold']),
            "best_threshold_recall": float(best_recall['threshold'])
        })
        
        # 5. Loggear métricas del mejor umbral (F1 como referencia)
        for metric, value in best_f1.items():
            if metric != 'threshold':
                mlflow.log_metric(f"best_f1_{metric}", float(value))
        
        # 6. Loggear tags
        if "tags" in config["mlflow"]:
            mlflow.set_tags(config["mlflow"]["tags"])
        
        mlflow.set_tag("tuning_type", "threshold")
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("eval_set", eval_set_name)
        
        # 7. Loggear modelo
        log_kwargs = {
            "sk_model": pipeline,
            "artifact_path": "model",
        }
        
        if "xgboost" in model_name.lower():
            log_kwargs["skops_trusted_types"] = [
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
                "xgboost.sklearn.XGBRegressor"
            ]
        
        mlflow.sklearn.log_model(**log_kwargs)
        
        # 8. Guardar resultados como artefacto
        mlflow.log_artifact(str(PROJECT_ROOT / "experiments" / "results" / f"{config['experiment']['name']}_thresholds.csv"))
        
        print(f"Resultados registrados en MLflow: {run_name}")


def main():
    """Comandos del script para ejecutar desde la terminal."""
    parser = argparse.ArgumentParser(
        description="Búsqueda del mejor umbral de clasificación"
    )
    parser.add_argument(
        "experiment",
        help="Nombre del experimento YAML (ej: exp008)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar output detallado"
    )
    
    args = parser.parse_args()
    tune_threshold(args.experiment, verbose=args.verbose)


if __name__ == "__main__":
    main()
