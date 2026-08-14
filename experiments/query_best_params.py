"""
query_best_params.py — Consultar mejores hiperparámetros
========================================================

Script de solo lectura que consulta MLflow para mostrar los mejores
hiperparámetros encontrados en el tuning.

Flujo:
    1. tune_experiment.py: Busca hiperparámetros y guarda en MLflow
    2. query_best_params.py: Consulta MLflow y muestra resultados
    3. generate_experiment.py: Genera YAML del siguiente paso

Uso:
    # Listar todos los runs de tuning con SMOTE
    python experiments/query_best_params.py --list --smote

    # Mostrar mejores parámetros del run ganador
    python experiments/query_best_params.py --smote

    # Consultar un run específico por ID
    python experiments/query_best_params.py --run-id abc123 --smote
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import yaml


# ============================================================
# Constantes
# ============================================================
EXPERIMENT_NAME = "home_credit_default"

def get_tuning_runs(smote: bool = True):
    """
    Busca todos los runs de tuning en MLflow.
    
    Args:
        smote: Si True, filtra solo runs con SMOTE habilitado
    
    Returns:
        DataFrame con los runs encontrados
    """
    filter_parts = ["tags.tuning_method = 'grid_search_cv'"]
    
    if smote:
        filter_parts.append("tags.smote_enabled = 'True'")
    else:
        filter_parts.append("tags.smote_enabled = 'False'")
    
    filter_string = " AND ".join(filter_parts)
    
    runs = mlflow.search_runs(
        filter_string=filter_string,
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.recall DESC"]
    )
    
    return runs


def get_best_params(run_id: str):
    """
    Obtiene los mejores parámetros de un run específico.
    
    Args:
        run_id: ID del run en MLflow
    
    Returns:
        Dict con los parámetros del modelo
    """
    run = mlflow.get_run(run_id)
    
    # Extraer solo los parámetros del modelo (empiezan con 'model__')
    model_params = {}
    for key, value in run.data.params.items():
        if key.startswith("model__"):
            # Quitar prefijo 'model__'
            param_name = key.replace("model__", "")
            # Convertir tipos
            if value.isdigit():
                model_params[param_name] = int(value)
            else:
                try:
                    model_params[param_name] = float(value)
                except ValueError:
                    model_params[param_name] = value
    
    return model_params, run.data.params, run.data.metrics


def print_params(model_params: dict, all_params: dict, metrics: dict, run_id: str):
    """Imprime los parámetros de forma formateada."""
    print("\n" + "=" * 70)
    print(f"  MEJORES HIPERPARÁMETROS (run: {run_id[:8]})")
    print("=" * 70)
    
    print("\nMétricas del run:")
    print(f"  - Recall (CV):     {metrics.get('cv_best_score', 'N/A')}")
    print(f"  - Recall (test):   {metrics.get('recall', 'N/A')}")
    print(f"  - F1 (test):       {metrics.get('f1', 'N/A')}")
    print(f"  - ROC AUC (test):  {metrics.get('roc_auc', 'N/A')}")
    
    print("\nMejores parámetros del modelo:")
    for param, value in model_params.items():
        print(f"  - {param}: {value}")
    
    print("\n" + "=" * 70)


def list_runs(smote: bool = True):
    """Lista todos los runs de tuning disponibles."""
    runs = get_tuning_runs(smote)
    
    if runs.empty:
        print("\nNo se encontraron runs de tuning.")
        return
    
    smote_label = "CON SMOTE" if smote else "SIN SMOTE"
    print(f"\n{'=' * 70}")
    print(f"  RUNS DE TUNING {smote_label}")
    print(f"{'=' * 70}")
    print(f"\n{'Run ID':<12} {'Run Name':<45} {'Recall':>8} {'F1':>8}")
    print("-" * 77)
    
    for _, row in runs.iterrows():
        run_id = row['run_id'][:8]
        run_name = row.get('run_name', 'N/A')[:44]
        recall = row.get('metrics.recall', 'N/A')
        f1 = row.get('metrics.f1', 'N/A')
        
        recall_str = f"{recall:.4f}" if isinstance(recall, (int, float)) else recall
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else f1
        
        print(f"  {run_id:<12} {run_name:<45} {recall_str:>8} {f1_str:>8}")
    
    print(f"\nTotal: {len(runs)} runs encontrados")


def main():
    parser = argparse.ArgumentParser(
        description="Consultar mejores hiperparámetros y generar experimento"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Listar todos los runs de tuning"
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Filtrar runs con SMOTE habilitado"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID específico (default: el mejor por recall)"
    )
    
    args = parser.parse_args()
    
    # Modo: listar runs
    if args.list:
        list_runs(smote=args.smote)
        return
    
    # Modo: obtener mejores params
    runs = get_tuning_runs(smote=args.smote)
    
    if runs.empty:
        print("\n No se encontraron runs de tuning.")
        print("   Asegurar que se ejecuto un tune_experiment.py primero.")
        return
    
    # Seleccionar run
    if args.run_id:
        # Buscar run específico
        run_row = runs[runs['run_id'].str.startswith(args.run_id)]
        if run_row.empty:
            print(f"\n No se encontró run con ID: {args.run_id}")
            return
        run_id = run_row.iloc[0]['run_id']
    else:
        # Usar el mejor por recall
        run_id = runs.iloc[0]['run_id']
    
    # Obtener parámetros
    model_params, all_params, metrics = get_best_params(run_id)
    
    # Imprimir parámetros
    print_params(model_params, all_params, metrics, run_id)
    print()


if __name__ == "__main__":
    main()
