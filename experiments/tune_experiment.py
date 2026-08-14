"""
tune_experiment.py — Tuning de hiperparámetros con Cross-Validation
====================================================================

Este script ejecuta tuning de hiperparámetros usando GridSearchCV
o RandomizedSearchCV con validación cruzada.

Uso:
    python experiments/tune_experiment.py exp005_tuning
    python experiments/tune_experiment.py exp005_tuning --verbose
    python experiments/tune_experiment.py --list
"""

import sys
import argparse
from pathlib import Path

# Agregar raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.data import load_config, load_dataset, load_features
from src.features import prepare_data
from src.models import create_pipeline, get_model_name


def setup_mlflow(config: dict):
    """Configura el experimento en MLflow."""
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)


def create_search_object(config: dict, pipeline):
    """
    Crea el objeto de búsqueda (GridSearchCV o RandomizedSearchCV).
    
    Args:
        config: Diccionario de configuración del experimento
        pipeline: Pipeline a optimizar
    
    Returns:
        GridSearchCV o RandomizedSearchCV
    """
    tuning_config = config["tuning"]
    method = tuning_config["method"]
    param_grid = tuning_config["param_grid"]
    
    # Parámetros comunes
    search_params = {
        "estimator": pipeline,
        "param_grid": param_grid,
        "cv": tuning_config["cv"],
        "scoring": tuning_config["scoring"],
        "n_jobs": tuning_config.get("n_jobs", -1),
        "verbose": tuning_config.get("verbose", 1),
        "return_train_score": True
    }
    
    # Crear objeto de búsqueda
    if method == "random":
        search = RandomizedSearchCV(
            **search_params,
            n_iter=tuning_config.get("n_iter", 50),
            random_state=config["data"]["random_state"]
        )
    else:
        search = GridSearchCV(**search_params)
    
    return search


def log_to_mlflow(config: dict, search, best_metrics: dict, cv_results_df: pd.DataFrame):
    """
    Registra todos los resultados del tuning en MLflow.
    
    Args:
        config: Diccionario de configuración del experimento
        search: Objeto GridSearchCV/RandomizedSearchCV entrenado
        best_metrics: Métricas del mejor modelo en test
        cv_results_df: DataFrame con resultados de CV
    """
    # Determinar sufijo SMOTE para el run name
    smote_config = config.get('smote', {})
    smote_suffix = "_smote" if smote_config.get('enabled', False) else ""
    
    with mlflow.start_run(run_name=f"tuning_{config['experiment']['name']}{smote_suffix}"):
        # 1. Loggear mejores hiperparámetros
        mlflow.log_params(search.best_params_)
        
        # 2. Loggear métricas de CV
        mlflow.log_metric("cv_best_score", search.best_score_)
        mlflow.log_metric("cv_std", cv_results_df.loc[search.best_index_, "std_test_score"])
        
        # 3. Loggear métricas de test
        mlflow.log_metrics(best_metrics)
        
        # 4. Loggear configuración del tuning
        mlflow.log_params({
            "tuning_method": config["tuning"]["method"],
            "cv_folds": config["tuning"]["cv"],
            "scoring": config["tuning"]["scoring"],
            "n_combinations": len(cv_results_df),
            "features_file": config["data"]["features_file"]
        })
        
        # 5. Loggear configuración SMOTE si está habilitada
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
        
        # 6. Loggear tags
        if "tags" in config["mlflow"]:
            mlflow.set_tags(config["mlflow"]["tags"])
        
        # 7. Loggear modelo mejorado
        model_name = config["model"]["name"].lower()
        
        log_model_kwargs = {
            "sk_model": search.best_estimator_,
            "artifact_path": "best_model"
        }
        
        if "xgboost" in model_name:
            log_model_kwargs["skops_trusted_types"] = [
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
                "xgboost.sklearn.XGBRegressor"
            ]
        
        mlflow.sklearn.log_model(**log_model_kwargs)
        
        # 7. Loggear resultados de CV como artifact
        cv_results_path = PROJECT_ROOT / config["output"]["cv_results_path"]
        cv_results_path.parent.mkdir(parents=True, exist_ok=True)
        cv_results_df.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(str(cv_results_path))
        
        print(f"\nResultados registrados en MLflow")
        print(f"  Experimento: {config['mlflow']['experiment_name']}")
        print(f"  Best CV Score: {search.best_score_:.4f}")


def print_search_summary(search, config: dict):
    """
    Imprime un resumen formateado de los resultados del tuning.
    """
    print("\n" + "=" * 70)
    print("  RESULTADOS DEL TUNING")
    print("=" * 70)
    
    print(f"\nMétodo: {config['tuning']['method'].upper()}")
    print(f"Cross-Validation: {config['tuning']['cv']} folds")
    print(f"Scoring: {config['tuning']['scoring']}")
    
    print(f"\nCombinaciones probadas: {len(search.cv_results_['params'])}")
    print(f"Mejor CV Score: {search.best_score_:.4f}")
    
    print(f"\nMejores hiperparámetros:")
    print("-" * 40)
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    print("\n" + "=" * 70)


def tune_experiment(experiment_name: str, verbose: bool = True):
    """
    Función principal: ejecuta tuning de hiperparámetros.
    
    Args:
        experiment_name: Nombre del experimento (ej: "exp005_tuning")
        verbose: Si True, muestra output detallado
    """
    print("\n" + "=" * 70)
    print(f"  TUNING: {experiment_name.upper()}")
    print("=" * 70)
    
    # 1. Cargar configuración
    config = load_config(experiment_name)
    
    # 2. Cargar y preparar datos
    df = load_dataset(config)
    features = load_features(config)
    X_train, X_test, y_train, y_test = prepare_data(config, df, features)
    
    # 3. Crear pipeline
    pipeline = create_pipeline(config)
    
    # 4. Crear objeto de búsqueda
    search = create_search_object(config, pipeline)
    
    # 5. Ejecutar tuning
    print(f"\nEjecutando {config['tuning']['method'].upper()}...")
    print(f"Entrenando todos los modelos...\n")
    
    search.fit(X_train, y_train)
    
    # 6. Mostrar resumen
    if verbose:
        print_search_summary(search, config)
    
    # 7. Evaluar mejor modelo en test
    from src.models import evaluate_model
    best_metrics = evaluate_model(search.best_estimator_, X_test, y_test, config, verbose=verbose)
    
    # 8. Guardar resultados de CV
    cv_results_df = pd.DataFrame(search.cv_results_)
    
    # 9. Loggear en MLflow
    setup_mlflow(config)
    log_to_mlflow(config, search, best_metrics, cv_results_df)
    
    print("\n" + "=" * 70)
    print(f"  TUNING COMPLETADO: {experiment_name.upper()}")
    print("=" * 70)
    
    return search, best_metrics


def list_tuning_experiments():
    """Lista todos los experimentos de tuning disponibles."""
    config_dir = PROJECT_ROOT / "configs" / "experiments"
    
    print("\nExperimentos de tuning disponibles:")
    print("-" * 50)
    
    for yaml_file in sorted(config_dir.glob("*tuning*.yaml")):
        config = load_config(yaml_file.stem)
        method = config.get("tuning", {}).get("method", "N/A")
        cv = config.get("tuning", {}).get("cv", "N/A")
        n_params = len(config.get("tuning", {}).get("param_grid", {}))
        
        print(f"  {yaml_file.stem}")
        print(f"    Método: {method} | CV: {cv} folds | Parámetros: {n_params}")
    
    print()


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Tuning de hiperparámetros con Cross-Validation para mejorar el baseline del modelo. Los experimentos se definen en archivos YAML en configs/experiments"
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="Nombre del experimento de tuning (ej: exp005_tuning)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Listar experimentos de tuning disponibles"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar output detallado"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_tuning_experiments()
    elif args.experiment:
        tune_experiment(args.experiment, verbose=args.verbose)
    else:
        parser.print_help()
        print("\nEjemplos de uso:")
        print("  python experiments/tune_experiment.py exp005_tuning")
        print("  python experiments/tune_experiment.py --list")
        print("  python experiments/tune_experiment.py exp005_tuning --verbose")


if __name__ == "__main__":
    main()
