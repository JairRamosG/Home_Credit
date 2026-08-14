"""
run_experiment.py — Orquestador principal de experimentos
==========================================================

Ejecuta el pipeline completo de un experimento:
1. Carga la configuración desde YAML
2. Carga y prepara los datos
3. Entrena el modelo
4. Evalúa el rendimiento en TEST SET
5. Registra todo en MLflow

IMPORTANTE: Este script evalúa en el TEST SET (evaluación final).
Para threshold tuning, usar tune_threshold.py (usa validation set).

Uso:
    python experiments/run_experiment.py exp001
    python experiments/run_experiment.py exp002
    python experiments/run_experiment.py exp009  # evaluación final con threshold
    python experiments/run_experiment.py --all
"""

import sys
import argparse
from pathlib import Path

# Agregar raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import mlflow
import mlflow.sklearn
from src.data import load_experiment
from src.features import prepare_data
from src.models import train_model, evaluate_model, get_model_name


def setup_mlflow(config: dict):
    """
    Configura el experimento en MLflow.
    
    Args:
        config: Diccionario de configuración del experimento
    """
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)


def log_to_mlflow(config: dict, metrics: dict, pipeline, run_name: str):
    """
    Registra todos los resultados del experimento en MLflow.
    
    Args:
        config: Diccionario de configuración del experimento
        metrics: Diccionario con las métricas
        pipeline: Pipeline entrenado
        run_name: Nombre del run
    """
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
        
        # 3. Loggear threshold si está configurado
        threshold = config.get("evaluation", {}).get("threshold", None)
        if threshold is not None:
            mlflow.log_param("threshold", float(threshold))
            mlflow.log_param("eval_set", "test")
        
        # 4. Loggear configuración SMOTE si está habilitada
        smote_config = config.get('smote', {})
        if smote_config.get('enabled', False):
            mlflow.log_params({
                "smote_enabled": True,
                "smote_sampling_strategy": smote_config.get('sampling_strategy', 0.5),
                "smote_k_neighbors": smote_config.get('k_neighbors', 5),
                "smote_random_state": smote_config.get('random_state', 42)
            })
        else:
            mlflow.log_param("smote_enabled", False)
        
        # 5. Loggear métricas
        mlflow.log_metrics(metrics)
        
        # 6. Loggear tags
        if "tags" in config["mlflow"]:
            mlflow.set_tags(config["mlflow"]["tags"])
        
        # 7. Loggear modelo
        model_name = config["model"]["name"].lower()
        
        # XGBoost necesita trusted types para skops
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
        
        # 7. Loggear features utilizadas
        features = config["data"]["features_file"]
        mlflow.log_param("features_count", len(open(PROJECT_ROOT / "data/processed" / features).readlines()))
        
        print(f"\nResultados registrados en MLflow")
        print(f"  Experimento: {config['mlflow']['experiment_name']}")
        print(f"  Run: {run_name}")


def run_experiment(experiment_name: str, verbose: bool = True):
    """
    Ejecuta un experimento completo.
    
    IMPORTANTE: Evalúa en el TEST SET (evaluación final).
    
    Args:
        experiment_name: Nombre del experimento (ej: "exp001")
        verbose: Si True, muestra output detallado
    
    Returns:
        Diccionario con las métricas del experimento
    """
    print("\n" + "=" * 70)
    print(f"  EXPERIMENTO: {experiment_name.upper()}")
    print("=" * 70)
    
    # 1. Cargar configuración y datos
    config, df, features = load_experiment(experiment_name)
    
    # 2. Preparar datos (con 3-way split si validation_size está definido)
    result = prepare_data(config, df, features)
    
    # 3. Manejar según el tipo de split
    if len(result) == 6:
        # 3-way split: train/val/test
        X_train, X_val, X_test, y_train, y_val, y_test = result
        
        # Para evaluación final: combinar Train + Validation = Nuevo Train (80%)
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        print(f"\n Evaluando en TEST SET (3-way split)")
        print(f"  - Train: {X_train_full.shape[0]:,} rows (80%)")
        print(f"  - Test: {X_test.shape[0]:,} rows (20%)")
    else:
        # 2-way split: train/test
        X_train_full, X_test, y_train_full, y_test = result
        print(f"\n Evaluando en TEST SET (2-way split)")
    
    # 4. Entrenar modelo (en train completo = train + val si es 3-way)
    pipeline = train_model(config, X_train_full, y_train_full)
    
    # 5. Evaluar modelo en TEST SET
    metrics = evaluate_model(pipeline, X_test, y_test, config, verbose=verbose)
    
    # 6. Registrar en MLflow
    model_name = get_model_name(config)
    smote_config = config.get('smote', {})
    smote_suffix = "_smote" if smote_config.get('enabled', False) else ""
    run_name = f"{experiment_name}_{model_name}{smote_suffix}"
    
    setup_mlflow(config)
    log_to_mlflow(config, metrics, pipeline, run_name)
    
    print("\n" + "=" * 70)
    print(f"  EXPERIMENTO COMPLETADO: {experiment_name.upper()}")
    print("=" * 70)
    
    return metrics


def discover_experiments():
    config_dir = Path("configs/experiments")
    experiments = []
    for yaml_file in config_dir.rglob("*.yaml"):
        # Obtener nombre relativo sin extensión
        # ej: baseline/exp001 o tuning/exp005_tuning
        rel_path = yaml_file.relative_to(config_dir)
        experiments.append(str(rel_path.with_suffix("")))
    return sorted(experiments)


def run_all_experiments(verbose: bool = False):
    """
    Ejecuta todos los experimentos disponibles.
    
    Args:
        verbose: Si True, muestra output detallado para cada uno
    """
    experiments = discover_experiments()
    results = {}
    
    print("\n" + "=" * 70)
    print("  EJECUTANDO TODOS LOS EXPERIMENTOS")
    print("=" * 70)
    
    for exp_name in experiments:
        try:
            metrics = run_experiment(exp_name, verbose=verbose)
            results[exp_name] = metrics
        except Exception as e:
            print(f"\n Error en {exp_name}: {e}")
            results[exp_name] = None
    
    # Resumen final
    print("\n" + "=" * 70)
    print("  RESUMEN DE RESULTADOS")
    print("=" * 70)
    
    for exp_name, metrics in results.items():
        if metrics:
            print(f"\n{exp_name}:")
            print(f"  - ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"  - F1: {metrics['f1']:.4f}")
            print(f"  - Recall: {metrics['recall']:.4f}")
        else:
            print(f"\n{exp_name}: ERROR")
    
    print("\n" + "=" * 70)
    print("  Ver resultados detallados en: mlflow ui")
    print("=" * 70)
    
    return results


def main():
    """Comandos del script para ejecutar los experimentos desde la terminal."""
    parser = argparse.ArgumentParser(
        description="Ejecutar experimentos de Home Credit Default Risk desde los archivos " \
        "de configuración YAML en configs/experiments"
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="Nombre del experimento (ej: exp001) o 'all' para ejecutar todos"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ejecutar todos los experimentos"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar output detallado"
    )
    
    args = parser.parse_args()
    
    if args.all or args.experiment == "all":
        run_all_experiments(verbose=args.verbose)
    elif args.experiment:
        run_experiment(args.experiment, verbose=args.verbose)
    else:
        parser.print_help()
        print("\nEjemplos de uso:")
        print("  python experiments/run_experiment.py exp001")
        print("  python experiments/run_experiment.py --all")
        print("  python experiments/run_experiment.py exp002 --verbose")


if __name__ == "__main__":
    main()
