"""
train.py — Módulo de entrenamiento de modelos
===============================================

Funciones para:
1. Crear modelos dinámicamente desde configuración YAML
2. Envolver en Pipeline con StandardScaler
3. Entrenar modelos
4. Soporte configurable para SMOTE (imblearn)
"""

import importlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def create_model(config: dict):
    """
    Crea un modelo dinámicamente desde la configuración YAML.
    
    Args:
        config: Diccionario de configuración del experimento
    
    Returns:
        Instancia del modelo sin entrenar
    
    Ejemplo:
        # En el YAML:
        # model:
        #   name: "sklearn.linear_model.LogisticRegression"
        #   params: {C: 1.0, penalty: "l2"}
        
        model = create_model(config)
        # → LogisticRegression(C=1.0, penalty='l2')
    """
    model_name = config["model"]["name"]
    params = config["model"]["params"]
    
    # Importar dinámicamente: "sklearn.linear_model.LogisticRegression"
    # → sklearn.linear_model → LogisticRegression
    module_path, class_name = model_name.rsplit(".", 1)
    
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    # Crear instancia con parámetros
    model = model_class(**params)
    
    return model


def create_smote(config: dict):
    """
    Crea una instancia de SMOTE desde la configuración YAML.
    
    Args:
        config: Diccionario de configuración del experimento
    
    Returns:
        Instancia de SMOTE o None si no está habilitado
    
    Raises:
        ImportError: Si imblearn no está instalado
    """
    smote_config = config.get('smote', {})
    
    if not smote_config.get('enabled', False):
        return None
    
    
    params = {
        'random_state': smote_config.get('random_state', 42),
        'sampling_strategy': smote_config.get('sampling_strategy', 0.5),
        'k_neighbors': smote_config.get('k_neighbors', 5)
    }
    
    return SMOTE(**params)


def create_pipeline(config: dict):
    """
    Crea un Pipeline configurable: StandardScaler + [SMOTE] + Modelo.
    
    Si smote.enabled=true en la config, usa imblearn.Pipeline con paso SMOTE.
    Si smote.enabled=false o no existe, usa sklearn.Pipeline estándar.
    
    Args:
        config: Diccionario de configuración del experimento
    
    Returns:
        Pipeline listo para entrenar (sklearn o imblearn según config)
    
    Ejemplo:
        pipeline = create_pipeline(config)
        pipeline.fit(X_train, y_train)
    """
    # Crear modelo
    model = create_model(config)
    
    # si SMOTE está habilitado
    smote = create_smote(config)
    
    if smote is not None:
        # Pipeline con SMOTE: Escalar + SMOTE + Modelo
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', smote),
            ('model', model)
        ])
    else:
        # Pipeline normal: Escalar + Modelo
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    
    return pipeline


def train_model(
    config: dict,
    X_train,
    y_train
) -> Pipeline:
    """
    Función principal: crea y entrena un modelo.
    
    Args:
        config: Diccionario de configuración del experimento
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
    
    Returns:
        Pipeline ya entrenado
    
    Ejemplo:
        config, df, features = load_experiment("exp001")
        X_train, X_test, y_train, y_test = prepare_data(config, df, features)
        pipeline = train_model(config, X_train, y_train)
    """
    print("=" * 60)
    print("Entrenando modelo")
    print("=" * 60)
    
    # Info del modelo
    model_name = get_model_name(config)
    params = config["model"]["params"]
    
    print(f"Modelo: {model_name}")
    print(f"Parámetros:")
    for key, value in params.items():
        print(f"  - {key}: {value}")
    
    # Info de SMOTE si está habilitado
    smote_config = config.get('smote', {})
    if smote_config.get('enabled', False):
        print(f"\nSMOTE habilitado:")
        print(f"  - sampling_strategy: {smote_config.get('sampling_strategy', 0.5)}")
        print(f"  - k_neighbors: {smote_config.get('k_neighbors', 5)}")
    
    # Crear pipeline
    pipeline = create_pipeline(config)
    print(f"\nPipeline creado: {pipeline}")

    
    # Entrenar
    print(f"\nEntrenando con {X_train.shape[0]:,} muestras")
    pipeline.fit(X_train, y_train)
    
    # Verificar que entrenó
    print("Modelo entrenado exitosamente")
    print("=" * 60)
    
    return pipeline


def get_model_name(config: dict) -> str:
    """
    Obtiene el nombre legible del modelo.
    
    Args:
        config: Diccionario de configuración del experimento
    
    Returns:
        Nombre del modelo (ej: "LogisticRegression")
    
    Ejemplo:
        name = get_model_name(config) 
    """
    return config["model"]["name"].split(".")[-1]


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
    
    # Cargar y preparar datos
    config, df, features = load_experiment("exp001")
    X_train, X_test, y_train, y_test = prepare_data(config, df, features)
    
    # Entrenar modelo
    pipeline = train_model(config, X_train, y_train)
    
    # Mostrar pipeline
    print("\nPipeline final:")
    print(pipeline)
