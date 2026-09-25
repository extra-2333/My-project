import os
import sys

import numpy as np
import tensorflow as tf

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pensamiento import DeepFullyConnectedNet, build_dataset, train_model
    from vision import VisionModel, listar_dataset_imagenes, extraer_features_vision
    from voz import AudioModel, TextToSpeechSynthesizer, listar_dataset_audio, extraer_features_audio
else:
    from .pensamiento import DeepFullyConnectedNet, build_dataset, train_model
    from .vision import VisionModel, listar_dataset_imagenes, extraer_features_vision
    from .voz import AudioModel, TextToSpeechSynthesizer, listar_dataset_audio, extraer_features_audio


NUM_CAPAS_OCULTAS = 50
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
CONGELAR_BACKBONE = True  


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_VISION_DIR = os.path.join(BASE_DIR, "dataset", "vision")
DATASET_VOZ_DIR = os.path.join(BASE_DIR, "dataset", "voz")
CARPETA_PESOS = os.path.join(BASE_DIR, "pesos_modelos")


def entrenar_pensamiento_generico(num_hidden_layers=NUM_CAPAS_OCULTAS):
    input_dim, hidden_dim, output_dim = 128, 256, 10
    n_samples = 2000

    X = np.random.randn(n_samples, input_dim).astype("float32")
    y = np.random.randint(0, output_dim, size=n_samples)

    dataset = build_dataset(X, y, batch_size=BATCH_SIZE)
    modelo = DeepFullyConnectedNet(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_hidden_layers=num_hidden_layers,
    )
    modelo.build(input_shape=(None, input_dim))

    print("GPUs disponibles:", tf.config.list_physical_devices("GPU"))
    print(f"Capas ocultas: {num_hidden_layers}")
    print(f"Parámetros entrenables: {modelo.count_params():,}")

    train_model(modelo, dataset, epochs=5, lr=LEARNING_RATE)
    return modelo


def entrenar_vision(dataset_dir=DATASET_VISION_DIR, num_hidden_layers=NUM_CAPAS_OCULTAS):
    _, _, clases = listar_dataset_imagenes(dataset_dir)
    print(f"[VISIÓN] Clases detectadas ({len(clases)}): {clases}")

    modelo = VisionModel(num_classes=len(clases), num_hidden_layers=num_hidden_layers)
    modelo.encoder.backbone.trainable = not CONGELAR_BACKBONE

    print("[VISIÓN] Extrayendo características de las imágenes...")
    X, y, clases = extraer_features_vision(dataset_dir, modelo.encoder)
    dataset = build_dataset(X, y, batch_size=BATCH_SIZE)

    train_model(modelo.head, dataset, epochs=EPOCHS, lr=LEARNING_RATE)

    os.makedirs(CARPETA_PESOS, exist_ok=True)
    ruta_pesos = os.path.join(CARPETA_PESOS, "vision_head.weights.h5")
    modelo.head.save_weights(ruta_pesos)
    print(f"[VISIÓN] Pesos guardados en: {ruta_pesos}")

    return modelo, clases


def cargar_vision(num_classes, num_hidden_layers=NUM_CAPAS_OCULTAS):
    modelo = VisionModel(num_classes=num_classes, num_hidden_layers=num_hidden_layers)
    modelo.head.build(input_shape=(None, modelo.encoder.feature_dim))
    modelo.head.load_weights(os.path.join(CARPETA_PESOS, "vision_head.weights.h5"))
    return modelo


def entrenar_voz(dataset_dir=DATASET_VOZ_DIR, num_hidden_layers=NUM_CAPAS_OCULTAS):
    _, _, clases = listar_dataset_audio(dataset_dir)
    print(f"[VOZ] Clases detectadas ({len(clases)}): {clases}")

    modelo = AudioModel(num_classes=len(clases), num_hidden_layers=num_hidden_layers)
    modelo.encoder.backbone.trainable = not CONGELAR_BACKBONE

    print("[VOZ] Extrayendo características de los audios...")
    X, y, clases = extraer_features_audio(dataset_dir, modelo.encoder)
    dataset = build_dataset(X, y, batch_size=BATCH_SIZE)

    train_model(modelo.head, dataset, epochs=EPOCHS, lr=LEARNING_RATE)

    os.makedirs(CARPETA_PESOS, exist_ok=True)
    ruta_pesos = os.path.join(CARPETA_PESOS, "voz_head.weights.h5")
    modelo.head.save_weights(ruta_pesos)
    print(f"[VOZ] Pesos guardados en: {ruta_pesos}")

    return modelo, clases


def cargar_voz(num_classes, num_hidden_layers=NUM_CAPAS_OCULTAS):
    modelo = AudioModel(num_classes=num_classes, num_hidden_layers=num_hidden_layers)
    modelo.head.build(input_shape=(None, modelo.encoder.feature_dim))
    modelo.head.load_weights(os.path.join(CARPETA_PESOS, "voz_head.weights.h5"))
    return modelo


def demo_hablar(texto):
    
    tts = TextToSpeechSynthesizer()
    ruta_salida = tts.synthesize(texto)
    print(f"[VOZ] Audio generado en: {ruta_salida}")
    return ruta_salida


if __name__ == "__main__":
    # 1) Prueba rápida de la arquitectura con datos sintéticos
    entrenar_pensamiento_generico()

    # 2) Entrenamiento real (descomenta cuando tu dataset esté listo)
    # entrenar_vision()
    # entrenar_voz()

    # 3) Ejemplo de habla (requiere pip install TensorFlowTTS soundfile)
    # demo_hablar("Hola, esta es una voz generada artificialmente.")
