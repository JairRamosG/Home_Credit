"""
selector.py — Módulo de selección de datos
==========================================

Funciones para:
1. Seleccionar features del DataFrame
2. Separar features (X) de target (y)
3. Dividir en train/validation/test

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Union


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
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
]:
    """
    Divide los datos en train/test o train/validation/test.
    
    Si config['data'] tiene 'validation_size', hace split en 3.
    Si no, hace split en 2 (comportamiento original).
    
    Args:
        X: DataFrame con las features
        y: Serie con el target
        config: Diccionario de configuración del experimento
    
    Returns:
        Si validation_size NO está definido:
            Tupla de (X_train, X_test, y_train, y_test)
        Si validation_size ESTÁ definido:
            Tupla de (X_train, X_val, X_test, y_train, y_val, y_test)
    
    Ejemplo:
        # Sin validation (comportamiento original)
        X_train, X_test, y_train, y_test = split_data(X, y, config)
        
        # Con validation
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)
    """
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    stratify = config["data"].get("stratify", False)
    validation_size = config["data"].get("validation_size", None)
    
    if validation_size is not None:
        # Split en 3: train + validation + test
        # Primero: separar test (test_size del total)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        
        # Después: separar train y validation del temporal
        # validation_size es la proporción del temporal que va a validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_temp if stratify else None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        # Split en 2: train + test (comportamiento original)
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
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
]:
    """
    Función principal: prepara todos los datos para entrenamiento.
    
    Ejecuta en orden:
    1. Selecciona features
    2. Separa target
    3. Divide en train/test o train/validation/test
    
    Args:
        config: Diccionario de configuración del experimento
        df: DataFrame completo
        features: Lista de features a usar
    
    Returns:
        Si validation_size NO está definido en config:
            Tupla de (X_train, X_test, y_train, y_test)
        Si validation_size ESTÁ definido en config:
            Tupla de (X_train, X_val, X_test, y_train, y_val, y_test)
    
    Ejemplo:
        config, df, features = load_experiment("exp001")
        
        # Sin validation
        X_train, X_test, y_train, y_test = prepare_data(config, df, features)
        
        # Con validation (si validation_size está en el YAML)
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(config, df, features)
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
    validation_size = config["data"].get("validation_size", None)
    
    if validation_size is not None:
        # Split en 3
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)
        
        print(f"\nSplit train/val/test:")
        print(f"  - Train: {X_train.shape[0]:,} muestras ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"  - Val:   {X_val.shape[0]:,} muestras ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"  - Test:  {X_test.shape[0]:,} muestras ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
        
        # Verificar distribución después del split
        if config["data"].get("stratify", False):
            train_ratio = y_train.mean()
            val_ratio = y_val.mean()
            test_ratio = y_test.mean()
            print(f"\nProporción del target (verificar stratify):")
            print(f"  - Train: {train_ratio*100:.2f}%")
            print(f"  - Val:   {val_ratio*100:.2f}%")
            print(f"  - Test:  {test_ratio*100:.2f}%")
        
        print("=" * 60)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        # Split en 2 (comportamiento original)
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
    result = prepare_data(config, df, features)
    
    # Verificar cuántos valores se devolvieron
    if len(result) == 6:
        X_train, X_val, X_test, y_train, y_val, y_test = result
        print("\n3-way split activado")
        print(f"Train: {X_train.shape}")
        print(f"Val: {X_val.shape}")
        print(f"Test: {X_test.shape}")
    else:
        X_train, X_test, y_train, y_test = result
        print("\n2-way split (original)")
        print(f"Train: {X_train.shape}")
        print(f"Test: {X_test.shape}")
