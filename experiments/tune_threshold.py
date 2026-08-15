"""
tune_threshold.py — Búsqueda del mejor umbral de clasificación
==============================================================

Entrena el modelo UNA vez y prueba múltiples umbrales en el validation set
para encontrar el que maximiza F1-score.

Genera tres gráficas profesionales:
    1. Matriz de Confusión (cm)
    2. Precision / Recall / F1 vs Umbral (PRF1)
    3. Train vs Validation F1 vs Umbral (trainval)

Uso:
    python experiments/tune_threshold.py exp005_umbral
    python experiments/tune_threshold.py exp005_umbral --verbose
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
import matplotlib.dates as mdates
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from src.data import load_experiment
from src.features import prepare_data
from src.models import train_model, get_model_name


# ============================================================
# Constantes
# ============================================================
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


# ============================================================
# Funciones auxiliares
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):
    """Calcula todas las métricas de evaluación."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }


def find_best_threshold(thresholds, metric_values):
    """Retorna el umbral con el valor máximo de una métrica."""
    idx = np.argmax(metric_values)
    return thresholds[idx], metric_values[idx]


def get_smote_suffix(config):
    """Obtiene sufijo SMOTE del config."""
    smote_config = config.get('smote', {})
    return "_smote" if smote_config.get('enabled', False) else "_baseline"


# ============================================================
# Gráficas 
# ============================================================

def create_confusion_matrix_plot(y_true, y_pred, threshold, experiment_name, set_name="Validation"):
    """
    Gráfica 1: Matriz de Confusión
    Layout: TP FN / FP TN (estándar de negocio)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Reorganizar a layout: [[TP, FN], [FP, TN]]
    cm_display = np.array([[tp, fn], [fp, tn]])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Crear matriz con colores
    im = ax.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    
    # Configurar ejes
    classes = ['Default (1)', 'No Default (0)']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicción', ylabel='Valor Real',
           title=f'Matriz de Confusión — {experiment_name}\n{set_name} Set (θ={threshold:.2f})')
    
    # Rotar labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Agregar valores en cada celda
    thresh = cm_display.max() / 2.
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            ax.text(j, i, format(cm_display[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black",
                    fontsize=18, fontweight='bold')
    
    # Agregar métricas como texto (fuera de la matriz)
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = (
        f'Accuracy:  {accuracy:.3f}\n'
        f'Precision: {precision:.3f}\n'
        f'Recall:    {recall:.3f}\n'
        f'F1-Score:  {f1:.3f}'
    )
    
    # Posicionar métricas a la derecha, centradas verticalmente
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))
    
    # Fecha
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    ax.text(0.98, 0.02, timestamp, transform=ax.transAxes,
            fontsize=8, color='gray', ha='right', va='bottom')
    
    plt.subplots_adjust(right=0.7)
    plt.tight_layout()
    return fig


def create_metrics_plot(thresholds, precision, recall, f1, 
                        best_threshold, best_f1, experiment_name):
    """
    Gráfica 2: Precision / Recall / F1 vs Umbral
    Muestra el trade-off entre métricas para elegir el umbral óptimo.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot de métricas
    ax.plot(thresholds, precision, color='#2196F3', linewidth=2, label='Precision', marker='o', markersize=3)
    ax.plot(thresholds, recall, color='#FF9800', linewidth=2, label='Recall', marker='s', markersize=3)
    ax.plot(thresholds, f1, color='#4CAF50', linewidth=2.5, label='F1-Score', marker='^', markersize=4)
    
    # Marcar el mejor umbral
    ax.axvline(x=best_threshold, color='#F44336', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.plot(best_threshold, best_f1, 'o', color='#F44336', markersize=12, zorder=5)
    ax.annotate(f'F1 Maximo\n(θ={best_threshold:.2f}, F1={best_f1:.3f})',
                xy=(best_threshold, best_f1),
                xytext=(best_threshold + 0.08, best_f1 - 0.05),
                fontsize=10, fontweight='bold', color='#F44336',
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5))
    
    # Configuracion
    ax.set_xlabel('Umbral de Decision', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Threshold Tuning — {experiment_name}\nPrecision / Recall / F1 vs Umbral',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='center left', fontsize=11, framealpha=0.9)
    ax.set_xlim([thresholds[0], thresholds[-1]])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Fecha
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    ax.text(0.98, 0.02, f'{timestamp}', transform=ax.transAxes,
            fontsize=8, color='gray', ha='right', va='bottom')
    
    plt.tight_layout()
    return fig


def create_overfitting_plot(thresholds, f1_train, f1_val, best_threshold, best_f1_val, experiment_name):
    """
    Gráfica 3: Train vs Validation F1 vs Umbral
    Detecta overfitting comparando curvas de entrenamiento y validacion.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot de curvas
    ax.plot(thresholds, f1_train, color='#9C27B0', linewidth=2, 
            label='Train F1', marker='o', markersize=3, linestyle='-')
    ax.plot(thresholds, f1_val, color='#E91E63', linewidth=2.5, 
            label='Validation F1', marker='^', markersize=4, linestyle='-')
    
    # Marcar el mejor umbral
    ax.axvline(x=best_threshold, color='#F44336', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.plot(best_threshold, best_f1_val, 'o', color='#F44336', markersize=12, zorder=5)
    ax.annotate(f'F1 Val Maximo\n(θ={best_threshold:.2f}, F1={best_f1_val:.3f})',
                xy=(best_threshold, best_f1_val),
                xytext=(best_threshold + 0.08, best_f1_val - 0.08),
                fontsize=10, fontweight='bold', color='#F44336',
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5))
    
    # Calcular gap (overfitting indicator)
    idx_best = np.where(thresholds == best_threshold)[0][0]
    gap = f1_train[idx_best] - f1_val[idx_best]
    
    # Sombreado del gap
    ax.fill_between(thresholds, f1_train, f1_val, alpha=0.1, color='#9C27B0')
    
    # Configuracion
    ax.set_xlabel('Umbral de Decision', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Deteccion de Overfitting — {experiment_name}\nTrain vs Validation F1',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='center left', fontsize=11, framealpha=0.9)
    ax.set_xlim([thresholds[0], thresholds[-1]])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Indicador de overfitting
    if gap > 0.1:
        overfit_text = f'Gap: {gap:.3f} (posible overfitting)'
        overfit_color = '#F44336'
    elif gap > 0.05:
        overfit_text = f'Gap: {gap:.3f} (overfitting leve)'
        overfit_color = '#FF9800'
    else:
        overfit_text = f'Gap: {gap:.3f} (generalizacion buena)'
        overfit_color = '#4CAF50'
    
    ax.text(0.02, 0.98, overfit_text, transform=ax.transAxes,
            fontsize=10, fontweight='bold', color=overfit_color,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=overfit_color, alpha=0.8))
    
    # Fecha
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    ax.text(0.98, 0.02, f'{timestamp}', transform=ax.transAxes,
            fontsize=8, color='gray', ha='right', va='bottom')
    
    plt.tight_layout()
    return fig


# ============================================================
# Funcion principal
# ============================================================

def tune_threshold(experiment_name: str, verbose: bool = True):
    """
    Busqueda del mejor umbral de clasificacion.
    
    Args:
        experiment_name: Nombre del experimento YAML (ej: "exp005_umbral")
        verbose: Si True, muestra output detallado
    
    Returns:
        DataFrame con resultados de todos los umbrales
    """
    print("\n" + "=" * 70)
    print(f"  THRESHOLD TUNING: {experiment_name.upper()}")
    print("=" * 70)
    
    # 1. Cargar configuracion y datos
    config, df, features = load_experiment(experiment_name)
    
    # 2. Preparar datos (con 3-way split si validation_size esta definido)
    result = prepare_data(config, df, features)
    
    # 3. Verificar que tenemos validation set
    if len(result) == 6:
        X_train, X_val, X_test, y_train, y_val, y_test = result
        print("\nUsando VALIDATION SET para threshold tuning")
        X_eval = X_val
        y_eval = y_val
        eval_set_name = "validation"
    else:
        print("\nWARNING: No hay validation set definido en el config")
        print("   Usando TEST SET")
        X_train, X_test, y_train, y_test = result
        X_eval = X_test
        y_eval = y_test
        eval_set_name = "test"
    
    # 4. Entrenar modelo UNA vez
    pipeline = train_model(config, X_train, y_train)
    
    # 5. Obtener probabilidades en entrenamiento y evaluacion
    print(f"\nGenerando probabilidades en train y {eval_set_name} set...")
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    y_prob_eval = pipeline.predict_proba(X_eval)[:, 1]
    
    # 6. Generar rango de umbrales (denso y suave)
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    print(f"Probando {len(thresholds)} umbrales: [0.01 a 0.99]")
    print("-" * 70)
    
    # 7. Evaluar cada umbral
    results = []
    
    for threshold in thresholds:
        # Validation set
        y_pred_eval = (y_prob_eval >= threshold).astype(int)
        eval_metrics = calculate_metrics(y_eval, y_pred_eval, y_prob_eval)
        
        # Training set
        y_pred_train = (y_prob_train >= threshold).astype(int)
        train_metrics = calculate_metrics(y_train, y_pred_train, y_prob_train)
        
        # Guardar resultados
        row = {'threshold': threshold}
        for metric, value in eval_metrics.items():
            row[f'val_{metric}'] = value
        for metric, value in train_metrics.items():
            row[f'train_{metric}'] = value
        
        results.append(row)
    
    # 8. Crear DataFrame con resultados
    df_results = pd.DataFrame(results)
    
    # 9. Extraer arrays para graficas
    thresholds_arr = df_results['threshold'].values
    precision_arr = df_results['val_precision'].values
    recall_arr = df_results['val_recall'].values
    f1_val_arr = df_results['val_f1'].values
    f1_train_arr = df_results['train_f1'].values
    
    # 10. Encontrar mejor umbral por metrica
    best_threshold_f1, best_f1_val = find_best_threshold(thresholds_arr, f1_val_arr)
    best_threshold_roc, best_roc = find_best_threshold(thresholds_arr, df_results['val_roc_auc'].values)
    best_threshold_precision, best_prec = find_best_threshold(thresholds_arr, precision_arr)
    best_threshold_recall, best_rec = find_best_threshold(thresholds_arr, recall_arr)
    
    # Indice del mejor F1 para obtener metricas del train
    idx_best = np.where(thresholds_arr == best_threshold_f1)[0][0]
    best_f1_train = f1_train_arr[idx_best]
    
    # Obtener predicciones con el mejor umbral para la matriz de confusion
    y_pred_best_eval = (y_prob_eval >= best_threshold_f1).astype(int)
    
    # 11. Mostrar resultados
    print("\n" + "=" * 70)
    print("MEJORES UMBRALES POR METRICA")
    print("=" * 70)
    print(f"\n{'Metrica':<15} {'Umbral':>10} {'Valor':>10}")
    print("-" * 40)
    print(f"{'F1-Score':<15} {best_threshold_f1:>10.2f} {best_f1_val:>10.4f}")
    print(f"{'ROC AUC':<15} {best_threshold_roc:>10.2f} {best_roc:>10.4f}")
    print(f"{'Precision':<15} {best_threshold_precision:>10.2f} {best_prec:>10.4f}")
    print(f"{'Recall':<15} {best_threshold_recall:>10.2f} {best_rec:>10.4f}")
    
    print(f"\n{'=' * 70}")
    print(f"  MEJOR UMBRAL SELECCIONADO: {best_threshold_f1:.2f}")
    print(f"  F1 Validation: {best_f1_val:.4f}")
    print(f"  F1 Train:      {best_f1_train:.4f}")
    print(f"  Gap:           {best_f1_train - best_f1_val:.4f}")
    print(f"{'=' * 70}")
    
    # 12. Crear y guardar graficas
    experiment_label = config['experiment']['name']
    smote_suffix = get_smote_suffix(config)
    
    # Extraer solo el nombre del experimento (sin prefijo de carpeta)
    # ej: "smote/exp005_umbral" → "exp005_umbral"
    output_name = experiment_name.split('/')[-1] if '/' in experiment_name else experiment_name
    
    # Asegurar que existe la carpeta de figuras
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Gráfica 1: Matriz de Confusión
    fig1 = create_confusion_matrix_plot(
        y_eval, y_pred_best_eval, best_threshold_f1,
        experiment_label, eval_set_name
    )
    path_fig1 = FIGURES_DIR / f"{output_name}{smote_suffix}_cm.png"
    fig1.savefig(path_fig1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"\nGráfica 1 (CM) guardada: {path_fig1}")
    
    # Gráfica 2: Precision / Recall / F1
    fig2 = create_metrics_plot(
        thresholds_arr, precision_arr, recall_arr, f1_val_arr,
        best_threshold_f1, best_f1_val, experiment_label
    )
    path_fig2 = FIGURES_DIR / f"{output_name}{smote_suffix}_PRF1.png"
    fig2.savefig(path_fig2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"Gráfica 2 (PRF1) guardada: {path_fig2}")
    
    # Gráfica 3: Overfitting (Train vs Val)
    fig3 = create_overfitting_plot(
        thresholds_arr, f1_train_arr, f1_val_arr,
        best_threshold_f1, best_f1_val, experiment_label
    )
    path_fig3 = FIGURES_DIR / f"{output_name}{smote_suffix}_trainval.png"
    fig3.savefig(path_fig3, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    print(f"Gráfica 3 (TrainVal) guardada: {path_fig3}")
    
    # 13. Guardar resultados en CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{output_name}{smote_suffix}_thresholds.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"CSV guardado: {csv_path}")
    
    # 14. Registrar en MLflow
    log_to_mlflow(
        config, df_results, pipeline, eval_set_name,
        best_threshold_f1, best_f1_val, best_f1_train,
        path_fig1, path_fig2, path_fig3, csv_path
    )
    
    print("\n" + "=" * 70)
    print(f"  THRESHOLD TUNING COMPLETADO")
    print("=" * 70)
    
    return df_results


# ============================================================
# MLflow Logging
# ============================================================

def log_to_mlflow(config, df_results, pipeline, eval_set_name,
                best_threshold, best_f1_val, best_f1_train,
                fig1_path, fig2_path, fig3_path, csv_path):
    """Registra los resultados del threshold tuning en MLflow."""
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)
    
    model_name = get_model_name(config)
    full_model_name = config["model"]["name"]
    
    # Determinar sufijo SMOTE
    smote_suffix = get_smote_suffix(config)
    
    run_name = f"{config['experiment']['name']}_threshold_tuning{smote_suffix}"
    
    with mlflow.start_run(run_name=run_name):
        # 1. Loggear parametros del modelo
        mlflow.log_params(config["model"]["params"])
        
        # 2. Loggear configuracion de datos
        mlflow.log_params({
            "dataset": config["data"]["dataset"],
            "features_file": config["data"]["features_file"],
            "test_size": config["data"]["test_size"],
            "random_state": config["data"]["random_state"],
            "validation_size": config["data"].get("validation_size", "N/A")
        })
        
        # 3. Loggear configuracion SMOTE si aplica
        smote_config = config.get('smote', {})
        if smote_config.get('enabled', False):
            mlflow.log_params({
                "smote_enabled": True,
                "smote_sampling_strategy": smote_config.get('sampling_strategy', 0.5),
                "smote_k_neighbors": smote_config.get('k_neighbors', 5)
            })
        else:
            mlflow.log_param("smote_enabled", False)
        
        # 4. Loggear mejor umbral y metricas
        mlflow.log_params({
            "best_threshold": best_threshold,
            "eval_set": eval_set_name,
            "threshold_method": "f1_maximization"
        })
        
        mlflow.log_metrics({
            "best_f1_validation": best_f1_val,
            "best_f1_train": best_f1_train,
            "f1_gap": best_f1_train - best_f1_val,
            "threshold_roc_auc": float(df_results.loc[df_results['val_f1'].idxmax(), 'val_roc_auc']),
            "threshold_precision": float(df_results.loc[df_results['val_f1'].idxmax(), 'val_precision']),
            "threshold_recall": float(df_results.loc[df_results['val_f1'].idxmax(), 'val_recall']),
        })
        
        # 5. Loggear tags
        if "tags" in config["mlflow"]:
            mlflow.set_tags(config["mlflow"]["tags"])
        
        mlflow.set_tag("tuning_type", "threshold")
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("eval_set", eval_set_name)
        
        # 6. Loggear modelo
        log_kwargs = {
            "sk_model": pipeline,
            "artifact_path": "model",
        }
        
        trusted_types = []
        if "xgboost" in full_model_name.lower():
            trusted_types.extend([
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
                "xgboost.sklearn.XGBRegressor"
            ])
        
        if smote_config.get('enabled', False):
            trusted_types.extend([
                "imblearn.pipeline.Pipeline",
                "imblearn.over_sampling._smote.base.SMOTE",
                "sklearn.metrics._dist_metrics.EuclideanDistance64",
                "sklearn.neighbors._kd_tree.KDTree"
            ])
        
        if trusted_types:
            log_kwargs["skops_trusted_types"] = trusted_types
        
        mlflow.sklearn.log_model(**log_kwargs)
        
        # 7. Loggear graficas y CSV como artefactos
        mlflow.log_artifact(str(fig1_path))
        mlflow.log_artifact(str(fig2_path))
        mlflow.log_artifact(str(fig3_path))
        mlflow.log_artifact(str(csv_path))
        
        print(f"Resultados registrados en MLflow: {run_name}")


# ============================================================
# CLI
# ============================================================

def main():
    """Comandos del script para ejecutar desde la terminal."""
    parser = argparse.ArgumentParser(
        description="Busqueda del mejor umbral de clasificacion"
    )
    parser.add_argument(
        "experiment",
        help="Nombre del experimento YAML (ej: exp005_umbral)"
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
