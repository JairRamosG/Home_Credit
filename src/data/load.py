"""
load.py — Módulo de carga de datos y configuración
====================================================

Funciones para cargar:
1. Archivos YAML de configuración
2. Dataset preprocesado
3. Listas de features desde archivos .txt
"""

import pandas as pd
import yaml
from pathlib import Path

# ============================================================
# Rutas base del proyecto
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent  # /home/jair/Proyectos/Home_credit
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"


def load_config(experiment_name: str) -> dict:
    """
    Carga un archivo YAML de configuración.
    
    Args:
        experiment_name: Nombre del experimento (ej: "exp001")
                        o ruta completa al YAML
    
    Returns:
        Diccionario con la configuración del experimento
    
    Ejemplo:
        config = load_config("exp001")
        config = load_config("configs/experiments/exp001.yaml")
    """
    # Si ya es una ruta completa
    if experiment_name.endswith(".yaml"):
        yaml_path = Path(experiment_name)
    else:
        # Construir ruta desde el nombre
        yaml_path = CONFIG_DIR / f"{experiment_name}.yaml"
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {yaml_path}")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"Configuración cargada: {yaml_path.name}")
    return config


def load_dataset(config: dict) -> pd.DataFrame:
    """
    Carga el dataset preprocesado desde CSV.
    
    Args:
        config: Diccionario de configuración del experimento
    
    Returns:
        DataFrame con el dataset completo
    
    Ejemplo:
        config = load_config("exp001")
        df = load_dataset(config)
    """
    dataset_name = config["data"]["dataset"]
    dataset_path = DATA_DIR / dataset_name
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    
    return df


def load_features(config: dict) -> list:
    """
    Carga la lista de features desde un archivo .txt.
    
    Args:
        config: Diccionario de configuración del experimento
    
    Returns:
        Lista de nombres de features
    
    Ejemplo:
        config = load_config("exp001")
        features = load_features(config)
        # ['DAYS_BIRTH', 'credit_approval_ratio', 'bureau_avg_days_credit', ...]
    """
    features_file = config["data"]["features_file"]
    features_path = DATA_DIR / features_file
    
    if not features_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de features: {features_path}")
    
    with open(features_path, "r", encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]
    
    print(f"Features cargadas: {len(features)} columnas desde {features_file}")
    
    return features


def load_experiment(experiment_name: str) -> tuple:
    """
    Carga todo lo necesario para un experimento: config, dataset y features.
    
    Args:
        experiment_name: Nombre del experimento (ej: "exp001")
    
    Returns:
        Tupla de (config, df, features)
    
    Ejemplo:
        config, df, features = load_experiment("exp001")

    """
    print("=" * 60)
    print(f"Cargando experimento: {experiment_name}")
    print("=" * 60)
    
    # Cargar configuración
    config = load_config(experiment_name)
    
    # Cargar dataset
    df = load_dataset(config)
    
    # Cargar features
    features = load_features(config)
    
    # Verificar que las features existen en el dataset
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features no encontradas en el dataset: {missing}")
    
    print("=" * 60)
    print("Experimento cargado exitosamente")
    print(f"  - Modelo: {config['model']['name']}")
    print(f"  - Features: {len(features)}")
    print(f"  - Muestras: {df.shape[0]:,}")
    print("=" * 60)
    
    return config, df, features


# ============================================================
# Función de prueba
# ============================================================
if __name__ == "__main__":
    # Probar con exp001
    config, df, features = load_experiment("exp001")
    
    print("\nPrimeras 5 features:")
    for i, f in enumerate(features[:5], 1):
        print(f"  {i}. {f}")
