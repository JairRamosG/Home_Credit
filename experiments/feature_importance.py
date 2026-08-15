"""
feature_importance.py — Análisis de importancia de features
==========================================================

Carga el mejor modelo desde MLflow y analiza la importancia de cada feature.
Genera una gráfica profesional y un CSV con los resultados.

Uso:
    python experiments/feature_importance.py --run-id ABC123
    python experiments/feature_importance.py --run-id ABC123 --top 20
    python experiments/feature_importance.py --run-id ABC123 --output csv
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
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn


# ============================================================
# Constantes
# ============================================================
EXPERIMENT_NAME = "home_credit_default"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


# ============================================================
# Funciones auxiliares
# ============================================================

def resolve_run_id(run_id: str) -> str:
    """
    Resuelve un run ID truncado (8 caracteres) al ID completo (32 caracteres).
    
    Args:
        run_id: ID del run (puede estar truncado)
    
    Returns:
        ID completo del run
    """
    # Si ya tiene 32 caracteres, es el ID completo
    if len(run_id) == 32:
        return run_id
    
    # Buscar runs que empiecen con el ID truncado
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        raise ValueError(f"Experimento '{EXPERIMENT_NAME}' no encontrado en MLflow")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"run_id LIKE '{run_id}%'"
    )
    
    if len(runs) == 0:
        raise ValueError(f"No se encontró ningún run con ID que empiece con: {run_id}")
    
    return runs[0].info.run_id


def load_model_from_run(run_id: str):
    """
    Carga el modelo y feature names desde un run de MLflow.
    
    Args:
        run_id: ID del run en MLflow (puede estar truncado)
    
    Returns:
        Tupla de (model, feature_names, run_data)
    """
    # Resolver ID truncado si es necesario
    full_run_id = resolve_run_id(run_id)
    
    run = mlflow.get_run(full_run_id)
    
    # Cargar modelo
    model_uri = f"runs:/{full_run_id}/best_model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Intentar obtener feature names de los tags o params
    feature_names = None
    if "features_file" in run.data.params:
        features_file = run.data.params["features_file"]
        features_path = PROJECT_ROOT / "data" / "processed" / features_file
        if features_path.exists():
            with open(features_path, "r") as f:
                feature_names = [line.strip() for line in f if line.strip()]
    
    return model, feature_names, run.data


def get_feature_importance(model, feature_names):
    """
    Extrae la importancia de features del modelo.
    
    Args:
        model: Modelo entrenado (XGBoost, RandomForest, etc.)
        feature_names: Lista de nombres de features
    
    Returns:
        DataFrame con features y su importancia
    """
    # Obtener importancia del modelo
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("El modelo no tiene attribute feature_importances_ o coef_")
    
    # Crear DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Agregar porcentaje acumulado
    df_importance['cumulative_pct'] = df_importance['importance'].cumsum() / df_importance['importance'].sum()
    
    return df_importance


# ============================================================
# Gráficas profesionales
# ============================================================

def create_importance_plot(df_importance, top_n, experiment_name, model):
    """
    Gráfica de barras horizontal con las top N features.
    """
    # Tomar top N
    df_top = df_importance.head(top_n).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colores
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_top)))
    
    # Barras horizontales
    bars = ax.barh(df_top['feature'], df_top['importance'], color=colors)
    
    # Agregar valores en cada barra
    for bar, val in zip(bars, df_top['importance']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    # Configuración
    ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance — {experiment_name}\n'
                f'(Modelo: {get_model_type(model)})',
                fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Fecha
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    ax.text(0.98, 0.02, timestamp, transform=ax.transAxes,
            fontsize=8, color='gray', ha='right', va='bottom')
    
    plt.tight_layout()
    return fig


def create_cumulative_plot(df_importance, experiment_name):
    """
    Gráfica de importancia acumulada.
    Muestra cuántas features necesitás para cubrir X% de la importancia.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Barras de importancia individual
    x = range(len(df_importance))
    ax1.bar(x, df_importance['importance'], color='#2196F3', alpha=0.6, label='Importancia individual')
    ax1.set_xlabel('Features (ordenadas por importancia)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Importancia', fontsize=12, fontweight='bold', color='#2196F3')
    ax1.tick_params(axis='y', labelcolor='#2196F3')
    
    # Línea de acumulado
    ax2 = ax1.twinx()
    ax2.plot(x, df_importance['cumulative_pct'], color='#F44336', linewidth=2.5, 
            label='Importancia acumulada')
    ax2.set_ylabel('Importancia Acumulada (%)', fontsize=12, fontweight='bold', color='#F44336')
    ax2.tick_params(axis='y', labelcolor='#F44336')
    ax2.set_ylim([0, 1.05])
    
    # Línea de referencia al 80%
    idx_80 = np.argmax(df_importance['cumulative_pct'] >= 0.8)
    ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=idx_80, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate(f'80% de importancia\n→ {idx_80 + 1} features',
                xy=(idx_80, 0.8), xytext=(idx_80 + 10, 0.6),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Configuración
    ax1.set_title(f'Importancia Acumulada — {experiment_name}\n'
                f'Numero de features',
                fontsize=14, fontweight='bold', pad=15)
    
    # Leyendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Fecha
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    ax1.text(0.98, 0.02, timestamp, transform=ax1.transAxes,
            fontsize=8, color='gray', ha='right', va='bottom')
    
    plt.tight_layout()
    return fig


def get_model_type(model):
    """
    Detecta el tipo de modelo.
    
    Args:
        model: Modelo o Pipeline
    
    Returns:
        Nombre del tipo de modelo
    """
    # Extraer modelo real si es Pipeline
    if hasattr(model, 'steps'):
        model = model.steps[-1][1]
    
    class_name = type(model).__name__
    
    # Mapear nombres comunes
    model_map = {
        'XGBClassifier': 'XGBoost',
        'XGBRegressor': 'XGBoost',
        'RandomForestClassifier': 'Random Forest',
        'RandomForestRegressor': 'Random Forest',
        'GradientBoostingClassifier': 'Gradient Boosting',
        'LGBMClassifier': 'LightGBM',
        'LGBMRegressor': 'LightGBM',
        'LogisticRegression': 'Logistic Regression',
        ' SVC': 'SVM',
        'LinearSVC': 'Linear SVM',
    }
    
    return model_map.get(class_name, class_name)


# ============================================================
# Función principal
# ============================================================

def analyze_feature_importance(run_id: str, top_n: int = 30, output_csv: bool = True):
    """
    Analiza la importancia de features de un modelo guardado en MLflow.
    
    Args:
        run_id: ID del run en MLflow
        top_n: Número de top features a mostrar
        output_csv: Si True, guarda los resultados en CSV
    """
    print("\n" + "=" * 70)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    # 1. Cargar modelo
    print(f"\nCargando modelo desde run: {run_id[:8]}...")
    model, feature_names, run_data = load_model_from_run(run_id)
    
    if feature_names is None:
        print("ERROR: No se encontraron los nombres de las features")
        return
    
    print(f"Features: {len(feature_names)}")
    print(f"Modelo: {run_data.params.get('model_name', 'N/A')}")
    
    # 2. Obtener importancia
    print("\nCalculando importancia...")
    df_importance = get_feature_importance(model, feature_names)
    
    # 3. Mostrar resultados
    print("\n" + "=" * 70)
    print(f"  TOP {top_n} FEATURES")
    print("=" * 70)
    print(f"\n{'#':<4} {'Feature':<40} {'Importancia':>12} {'Acumulado':>10}")
    print("-" * 70)
    
    for i, row in df_importance.head(top_n).iterrows():
        idx = df_importance.index.get_loc(i) + 1
        print(f"{idx:<4} {row['feature']:<40} {row['importance']:>12.6f} {row['cumulative_pct']:>10.1%}")
    
    # 4. Estadísticas
    print("\n" + "=" * 70)
    print("  ESTADÍSTICAS")
    print("=" * 70)
    
    total_features = len(df_importance)
    features_with_importance = (df_importance['importance'] > 0).sum()
    features_above_1pct = (df_importance['importance'] > 0.01).sum()
    idx_80 = np.argmax(df_importance['cumulative_pct'] >= 0.8) + 1
    
    print(f"\n  Total features: {total_features}")
    print(f"  Features con importancia > 0: {features_with_importance}")
    print(f"  Features con importancia > 1%: {features_above_1pct}")
    print(f"  Features para cubrir 80%: {idx_80}")
    print(f"  Importancia máxima: {df_importance['importance'].max():.6f}")
    print(f"  Importancia mínima: {df_importance['importance'].min():.6f}")
    
    # 5. Guardar gráficas
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Detectar si es SMOTE del run
    smote_suffix = "_smote" if run_data.params.get('smote_enabled', 'False') == 'True' else "_baseline"
    
    # Gráfica 1: Top N features
    fig1 = create_importance_plot(df_importance, top_n, f"Run {run_id[:8]}", model)
    path_fig1 = FIGURES_DIR / f"feature_importance{smote_suffix}_top{top_n}.png"
    fig1.savefig(path_fig1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"\nGrafica guardada: {path_fig1}")
    
    # Gráfica 2: Acumulado
    fig2 = create_cumulative_plot(df_importance, f"Run {run_id[:8]}")
    path_fig2 = FIGURES_DIR / f"feature_importance{smote_suffix}_cumulative.png"
    fig2.savefig(path_fig2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"Grafica acumulada guardada: {path_fig2}")
    
    # 6. Guardar CSV
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / f"feature_importance{smote_suffix}.csv"
        df_importance.to_csv(csv_path, index=False)
        print(f"CSV guardado: {csv_path}")
    
    print("\n" + "=" * 70)
    print("  ANÁLISIS COMPLETADO")
    print("=" * 70)
    
    return df_importance


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Análisis de importancia de features"
    )
    parser.add_argument(
        "--run-id", "-r",
        type=str,
        required=True,
        help="ID del run en MLflow"
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=30,
        help="Número de top features a mostrar (default: 30)"
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="No guardar CSV"
    )
    
    args = parser.parse_args()
    analyze_feature_importance(args.run_id, top_n=args.top, output_csv=not args.no_csv)


if __name__ == "__main__":
    main()
