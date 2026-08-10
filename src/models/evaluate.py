"""
evaluate.py — Módulo de evaluación de modelos
===============================================

Funciones para:
1. Hacer predicciones con el pipeline entrenado
2. Calcular métricas de evaluación
3. Mostrar reportes de clasificación
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

def predict(pipeline, X_test, threshold=None):
    """
    Genera predicciones de clase y probabilidades.
    
    Args:
        pipeline: Pipeline entrenado
        X_test: Features de test
        threshold: Umbral de clasificación (default: None = usar pipeline.predict)
                   Si se especifica, se usa predict_proba con este umbral
    
    Returns:
        Diccionario con y_pred y y_prob
    
    Ejemplo:
        # Sin threshold (comportamiento original)
        predictions = predict(pipeline, X_test)
        
        # Con threshold personalizado
        predictions = predict(pipeline, X_test, threshold=0.3)
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # Probabilidad de clase 1
    
    if threshold is not None:
        # Aplicar umbral personalizado
        y_pred = (y_prob >= threshold).astype(int)
    else:
        # Comportamiento original: usar pipeline.predict()
        y_pred = pipeline.predict(X_test)
    
    return {
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def calculate_metrics(y_test, y_pred, y_prob) -> dict:
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        y_test: Valores reales
        y_pred: Predicciones de clase
        y_prob: Probabilidades de clase 1
    
    Returns:
        Diccionario con las métricas
    
    Ejemplo:
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        # {'accuracy': 0.91, 'precision': 0.52, 'recall': 0.03, ...}
    """
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    return metrics


def evaluate_model(
    pipeline,
    X_test,
    y_test,
    config: dict,
    verbose: bool = True
) -> dict:
    """
    Función principal: evalúa un modelo entrenado.
    
    Args:
        pipeline: Pipeline entrenado
        X_test: Features de test
        y_test: Target de test
        config: Diccionario de configuración del experimento
        verbose: Si True, muestra el reporte completo
    
    Returns:
        Diccionario con las métricas
    
    Ejemplo:
        metrics = evaluate_model(pipeline, X_test, y_test, config)
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    """
    if verbose:
        print("=" * 60)
        print("Evaluando modelo")
        print("=" * 60)
    
    # 1. Obtener threshold del config (default: None = usar pipeline.predict)
    threshold = config.get("evaluation", {}).get("threshold", None)
    if threshold is not None:
        threshold = float(threshold)
        if verbose:
            print(f"Usando umbral de clasificación: {threshold}")
    
    # 2. Hacer predicciones
    predictions = predict(pipeline, X_test, threshold=threshold)
    y_pred = predictions['y_pred']
    y_prob = predictions['y_prob']
    
    # 3. Calcular métricas
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    if verbose:
        # 3. Mostrar métricas
        print("\nMétricas de evaluación:")
        print("-" * 40)
        for metric_name, value in metrics.items():
            print(f"  {metric_name:12s}: {value:.4f}")
        
        # 4. Mostrar reporte de clasificación
        print("\nReporte de clasificación:")
        print("-" * 40)
        print(classification_report(y_test, y_pred, 
                                    target_names=['Pagó (0)', 'Default (1)']))
        
        # 5. Mostrar matriz de confusión
        print("Matriz de confusión:")
        print("-" * 40)
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Predijo Pagó:    {cm[0][0]:>6,} verdaderos | {cm[0][1]:>6,} falsos")
        print(f"  Predijo Default: {cm[1][0]:>6,} falsos    | {cm[1][1]:>6,} verdaderos")
        
        print("=" * 60)
    
    return metrics


# ============================================================
# Función de prueba
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Agregar raíz del proyecto al path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.data import load_experiment
    from src.features import prepare_data
    from src.models import train_model
    
    # Cargar, preparar y entrenar
    config, df, features = load_experiment("exp001")
    X_train, X_test, y_train, y_test = prepare_data(config, df, features)
    pipeline = train_model(config, X_train, y_train)
    
    # Evaluar
    metrics = evaluate_model(pipeline, X_test, y_test, config)
