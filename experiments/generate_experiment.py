"""
generate_experiment.py — Generar configs de experimentos automáticamente
========================================================================

Lee el config de un experimento fuente (ej: tuning) + los mejores params
de MLflow y genera el YAML para el siguiente experimento (ej: threshold).

Flujo:
    1.- exp00X_tuning.yaml (2-way split)
    2.- query MLflow y da best_params_
    3.- exp00X_umbral.yaml (3-way split + best_params)

Uso:
    # Generar threshold desde tuning con SMOTE
    python experiments/generate_experiment.py --source exp00X_tuning --type threshold --smote

    # Generar threshold desde tuning sin SMOTE
    python experiments/generate_experiment.py --source exp00X_tuning --type threshold

    # Listar configs disponibles
    python experiments/generate_experiment.py --list
"""

import sys
import argparse
from pathlib import Path

# Agregar raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import yaml


# ============================================================
# Constantes
# ============================================================
EXPERIMENT_NAME = "home_credit_default"
CONFIGS_DIR = PROJECT_ROOT / "configs" / "experiments"
EXPERIMENTS_CONFIGS = {
    "smote": CONFIGS_DIR / "smote",
    "baseline": CONFIGS_DIR / "baseline",
}


# ============================================================
# Funciones auxiliares
# ============================================================

def load_yaml(path: Path) -> dict:
    """Lee un YAML y retorna el dict."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Path):
    """Guarda un dict como YAML."""
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def get_best_params_from_mlflow(source_name: str, smote: bool = True) -> dict:
    """
    Consulta MLflow y retorna los mejores parámetros del modelo.
    
    Args:
        source_name: Nombre del experimento fuente (ej: "exp00X_tuning")
        smote: Si True, filtra runs con SMOTE
    
    Returns:
        Dict con parámetros del modelo (sin prefijo 'model__')
    """
    filter_parts = ["tags.tuning_method = 'grid_search_cv'"]
    
    if smote:
        filter_parts.append("tags.smote_enabled = 'True'")
    else:
        filter_parts.append("tags.smote_enabled = 'False'")
    
    runs = mlflow.search_runs(
        filter_string=" AND ".join(filter_parts),
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.recall DESC"]
    )
    
    if runs.empty:
        raise ValueError(f"No se encontraron runs de tuning en MLflow")
    
    run_id = runs.iloc[0]['run_id']
    run = mlflow.get_run(run_id)
    
    # Extraer parámetros del modelo (sin prefijo 'model__')
    model_params = {}
    for key, value in run.data.params.items():
        if key.startswith("model__"):
            param_name = key.replace("model__", "")
            if value.isdigit():
                model_params[param_name] = int(value)
            else:
                try:
                    model_params[param_name] = float(value)
                except ValueError:
                    model_params[param_name] = value
    
    print(f"Mejores params obtenidos de MLflow (run: {run_id[:8]})")
    return model_params


def find_source_config(source_name: str, smote: bool = True) -> Path:
    """
    Busca el YAML del experimento fuente.
    
    Args:
        source_name: Nombre del experimento (ej: "exp00X_tuning")
        smote: Si True, busca en carpeta smote/
    
    Returns:
        Path al YAML encontrado
    """
    folder = "smote" if smote else "baseline"
    config_path = CONFIGS_DIR / folder / f"{source_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró: {config_path}")
    
    return config_path


# ============================================================
# Generadores de experimentos
# ============================================================

def generate_threshold_config(source_config: dict, model_params: dict, source_name: str, smote: bool = True) -> dict:
    """
    Genera config para threshold tuning a partir de un config de tuning.
    
    Diferencias con el config fuente:
    - 3-way split (agrega validation_size)
    - Sin sección tuning (GridSearchCV)
    - Con sección evaluation.thresholds
    - Parámetros del modelo = best_params del tuning
    """
    # Detectar número de experimento del nombre fuente
    # ej: "exp005_tuning" = "exp005"
    exp_number = source_name.split("_")[0]
        
    # --- experiment ---
    new_config = {
        "experiment": {
            "name": f"{exp_number}_umbral",
            "description": f"Ajuste de umbral — basado en {source_name}",
            "version": "1.0",
            "author": "Jair",
            "stage": "etapa_3",
            "base_experiment": source_name,
        },
        
        # --- data: 3-way split ---
        "data": {
            "dataset": source_config["data"]["dataset"],
            "features_file": source_config["data"]["features_file"],
            "target": source_config["data"]["target"],
            "id_column": source_config["data"]["id_column"],
            "test_size": source_config["data"]["test_size"],
            "validation_size": 0.2,  # NUEVO: 3-way split
            "random_state": source_config["data"]["random_state"],
            "stratify": source_config["data"].get("stratify", True),
        },
    }
    
    # --- smote (si aplica) ---
    if smote and "smote" in source_config:
        new_config["smote"] = source_config["smote"]
    
    # --- model: best_params del tuning ---
    new_config["model"] = {
        "name": source_config["model"]["name"],
        "description": "Modelo con mejores hiperparámetros del tuning",
        "params": {
            **model_params,  # best_params de MLflow
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "auc",
        }
    }
    
    # Mantener scale_pos_weight si existía en el fuente
    if "scale_pos_weight" in source_config.get("model", {}).get("params", {}):
        if "scale_pos_weight" not in new_config["model"]["params"]:
            new_config["model"]["params"]["scale_pos_weight"] = \
                source_config["model"]["params"]["scale_pos_weight"]
    
    # --- evaluation: thresholds ---
    new_config["evaluation"] = {
        "thresholds": {
            "min": 0.10,
            "max": 0.50,
            "step": 0.05,
        },
        "primary_metric": "recall",
    }
    
    # --- metrics ---
    new_config["metrics"] = source_config.get("metrics", {
        "primary": "recall",
        "all": [
            {"name": "accuracy", "description": "Exactitud general"},
            {"name": "precision", "description": "¿Cuántos predichos como default realmente lo son?"},
            {"name": "recall", "description": "¿Cuántos defaults reales detectamos?"},
            {"name": "f1", "description": "Balance precision/recall"},
            {"name": "roc_auc", "description": "Área bajo la curva ROC"},
        ]
    })
    
    # --- mlflow ---
    new_config["mlflow"] = {
        "experiment_name": EXPERIMENT_NAME,
        "tags": {
            **source_config.get("mlflow", {}).get("tags", {}),
            "tuning_method": "threshold_search",
            "eval_set": "validation",
            "stage": "etapa_3",
        }
    }
    
    # --- output ---
    new_config["output"] = {
        "save_model": True,
        "model_path": f"models/{exp_number}_umbral.pkl",
        "save_predictions": True,
        "predictions_path": f"experiments/results/{exp_number}_umbral_predictions.csv",
        "save_threshold_results": True,
        "threshold_results_path": f"experiments/results/{exp_number}_threshold_results.csv",
    }
    
    return new_config


# ============================================================
# CLI
# ============================================================

def list_configs():
    """Muestra los configs disponibles."""
    print(f"\n{'=' * 60}")
    print("  CONFIGS DISPONIBLES")
    print(f"{'=' * 60}")
    
    for folder, path in EXPERIMENTS_CONFIGS.items():
        print(f"\n {folder}/")
        if path.exists():
            for yaml_file in sorted(path.glob("*.yaml")):
                print(f"   - {yaml_file.stem}")
        else:
            print("   (vacío)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generar configs de experimentos automáticamente"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Listar configs disponibles"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        help="Config fuente (ej: exp00X_tuning)"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["threshold"],
        help="Tipo de experimento a generar"
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Usar config de carpeta smote/"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar YAML sin guardarlo"
    )
    
    args = parser.parse_args()
    
    # Modo: listar
    if args.list:
        list_configs()
        return
    
    # Validar argumentos
    if not args.source:
        parser.error("Se requiere --source ")
    
    if not args.type:
        parser.error("Se requiere --type threshold")
    
    # 1. Cargar config fuente
    source_path = find_source_config(args.source, smote=args.smote)
    print(f" Config fuente: {source_path}")
    source_config = load_yaml(source_path)
    
    # 2. Obtener mejores params de MLflow
    model_params = get_best_params_from_mlflow(args.source, smote=args.smote)
    
    # 3. Generar config según tipo
    if args.type == "threshold":
        new_config = generate_threshold_config(
            source_config, model_params, args.source, smote=args.smote
        )
    
    # 4. Determinar path de salida
    output_name = new_config["experiment"]["name"]
    folder = "smote" if args.smote else "baseline"
    output_path = CONFIGS_DIR / folder / f"{output_name}.yaml"
    
    # 5. Mostrar o guardar
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("  YAML GENERADO (dry-run, no se guardó)")
        print(f"{'=' * 60}\n")
        print(yaml.dump(new_config, default_flow_style=False, sort_keys=False))
    else:
        save_yaml(new_config, output_path)
        print(f"\nYAML generado: {output_path}")
        print(f"\nPróximo paso:")
        print(f"  python experiments/tune_threshold.py {output_name}")


if __name__ == "__main__":
    main()
