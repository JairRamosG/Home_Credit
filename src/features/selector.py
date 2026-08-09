"""
selector.py — Módulo de selección de datos
==========================================

Funciones para:
1. Seleccionar features del DataFrame
2. Separar features (X) de target (y)
3. Dividir en train/test

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def select_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Selecciona solo las columnas especificadas del DataFrame.
    
    Args:
        df: DataFrame completo
        features: Lista de nombres de columnas a seleccionar
    
    Returns:
        DataFrame solo con las columnas seleccionadas
    
    Ejemplo:
        X = select_features(df, ['DAYS_BIRTH', 'credit_approval_ratio'])
    """
    # Verificar que todas las features existen
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features no encontradas en el dataset: {missing}")
    
    X = df[features].copy()
    return X


def select_target(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Extrae la columna target del DataFrame.
    
    Args:
        df: DataFrame completo
        config: Diccionario de configuración del experimento
    
    Returns:
        Serie con la variable target
    
    Ejemplo:
        y = select_target(df, config)  # config['data']['target'] = 'TARGET'
    """
    target_col = config["data"]["target"]
    
    if target_col not in df.columns:
        raise ValueError(f"Columna target no encontrada: {target_col}")
    
    y = df[target_col].copy()
    return y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en train y test.
    
    Args:
        X: DataFrame con las features
        y: Serie con el target
        config: Diccionario de configuración del experimento
    
    Returns:
        Tupla de (X_train, X_test, y_train, y_test)
    
    Ejemplo:
        X_train, X_test, y_train, y_test = split_data(X, y, config)
    """
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    stratify = config["data"].get("stratify", False)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )
    
    return X_train, X_test, y_train, y_test


def prepare_data(
    config: dict,
    df: pd.DataFrame,
    features: list
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Función principal: prepara todos los datos para entrenamiento.
    
    Ejecuta en orden:
    1. Selecciona features
    2. Separa target
    3. Divide en train/test
    
    Args:
        config: Diccionario de configuración del experimento
        df: DataFrame completo
        features: Lista de features a usar
    
    Returns:
        Tupla de (X_train, X_test, y_train, y_test)
    
    Ejemplo:
        config, df, features = load_experiment("exp001")
        X_train, X_test, y_train, y_test = prepare_data(config, df, features)
    """
    print("=" * 60)
    print("Preparando datos")
    print("=" * 60)
    
    # 1. Seleccionar features
    X = select_features(df, features)
    print(f"Features seleccionadas: {X.shape[1]}")
    
    # 2. Separar target
    y = select_target(df, config)
    print(f"Target: {config['data']['target']}")
    print(f"Target balance: {y.value_counts(normalize=True).to_dict()}")

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    print(f"\nSplit train/test ({config['data']['test_size']*100:.0f}% test):")
    print(f"  - Train: {X_train.shape[0]:,} muestras")
    print(f"  - Test: {X_test.shape[0]:,} muestras")
    
    # Verificar distribución después del split
    if config["data"].get("stratify", False):
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        print(f"\nProporción del target (verificar stratify):")
        print(f"  - Train: {train_ratio*100:.2f}%")
        print(f"  - Test: {test_ratio*100:.2f}%")
    
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test


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
    
    # Cargar experimento
    config, df, features = load_experiment("exp001")
    
    # Preparar datos
    X_train, X_test, y_train, y_test = prepare_data(config, df, features)
    
    print("\nPrimeras 3 features (train):")
    print(X_train.head(3))
