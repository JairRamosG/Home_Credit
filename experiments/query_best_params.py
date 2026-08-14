"""
query_best_params.py — Consultar mejores hiperparámetros y generar experimento
===============================================================================

Consulta MLflow para obtener los mejores hiperparámetros de un tuning run
y genera automáticamente el YAML para el siguiente experimento (threshold tuning).

Flujo profesional:
  1. tune_experiment.py → busca hiperparámetros → guarda en MLflow
  2. query_best_params.py → consulta MLflow → genera YAML del siguiente paso
  3. tune_threshold.py → usa el YAML generado

Uso:
    # Generar YAML para threshold tuning (desde exp005_tuning con SMOTE)
    python experiments/query_best_params.py --next-threshold --smote

    # Solo imprimir parámetros (sin generar YAML)
    python experiments/query_best_params.py --smote

    # Listar todos los runs de tuning
    python experiments/query_best_params.py --list
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Agregar raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import yaml


# ============================================================
# Constantes
# ============================================================
EXPERIMENT_NAME = "home_credit_default"
OUTPUT_DIR = PROJECT_ROOT / "configs" / "experiments" / "smote"
TEMPLATE_PATH = PROJECT_ROOT / "configs" / "experiments" / "baseline" / "exp005_umbral.yaml"


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


def generate_threshold_yaml(model_params: dict, run_id: str, smote: bool = True):
    """
    Genera el YAML para threshold tuning usando los mejores parámetros.
    
    Args:
        model_params: Parámetros del modelo (sin prefijo 'model__')
        run_id: ID del run de tuning (para documentar)
        smote: Si True, incluye configuración SMOTE
    """
    # Leer template
    with open(TEMPLATE_PATH, 'r') as f:
        template = yaml.safe_load(f)
    
    # Actualizar nombre del experimento
    template['experiment']['name'] = 'exp005_umbral'
    template['experiment']['description'] = (
        f'Ajuste de umbral — basado en exp005_tuning (run: {run_id[:8]})'
    )
    template['experiment']['base_experiment'] = 'exp005_tuning'
    template['experiment']['version'] = '3.0'
    template['experiment']['author'] = 'Jair'
    
    # Actualizar parámetros del modelo con los mejores del tuning
    template['model']['params'].update(model_params)
    template['model']['params']['random_state'] = 42
    template['model']['params']['n_jobs'] = -1
    template['model']['params']['eval_metric'] = 'auc'
    
    # Mantener scale_pos_weight si existe
    if 'scale_pos_weight' not in template['model']['params']:
        template['model']['params']['scale_pos_weight'] = 11.38
    
    # Guardar YAML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"exp005_umbral.yaml"
    
    with open(output_path, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    return output_path


def print_params(model_params: dict, all_params: dict, metrics: dict, run_id: str):
    """Imprime los parámetros de forma formateada."""
    print("\n" + "=" * 70)
    print(f"  MEJORES HIPERPARÁMETROS (run: {run_id[:8]})")
    print("=" * 70)
    
    print("\n📊 Métricas del run:")
    print(f"  - Recall (CV):     {metrics.get('cv_best_score', 'N/A')}")
    print(f"  - Recall (test):   {metrics.get('recall', 'N/A')}")
    print(f"  - F1 (test):       {metrics.get('f1', 'N/A')}")
    print(f"  - ROC AUC (test):  {metrics.get('roc_auc', 'N/A')}")
    
    print("\n🔧 Mejores parámetros del modelo:")
    for param, value in model_params.items():
        print(f"  - {param}: {value}")
    
    print("\n" + "=" * 70)


def list_runs(smote: bool = True):
    """Lista todos los runs de tuning disponibles."""
    runs = get_tuning_runs(smote)
    
    if runs.empty:
        print("\n⚠️  No se encontraron runs de tuning.")
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
        "--next-threshold",
        action="store_true",
        help="Generar YAML para threshold tuning"
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
        print("\n⚠️  No se encontraron runs de tuning.")
        print("   Asegurate de haber ejecutado tune_experiment.py primero.")
        return
    
    # Seleccionar run
    if args.run_id:
        # Buscar run específico
        run_row = runs[runs['run_id'].str.startswith(args.run_id)]
        if run_row.empty:
            print(f"\n❌ No se encontró run con ID: {args.run_id}")
            return
        run_id = run_row.iloc[0]['run_id']
    else:
        # Usar el mejor por recall
        run_id = runs.iloc[0]['run_id']
    
    # Obtener parámetros
    model_params, all_params, metrics = get_best_params(run_id)
    
    # Imprimir parámetros
    print_params(model_params, all_params, metrics, run_id)
    
    # Modo: generar YAML para threshold tuning
    if args.next_threshold:
        output_path = generate_threshold_yaml(model_params, run_id, smote=args.smote)
        print(f"\n✅ YAML generado: {output_path}")
        print(f"\nPróximo paso:")
        print(f"  python experiments/tune_threshold.py exp005_umbral")
    
    print()


if __name__ == "__main__":
    main()
