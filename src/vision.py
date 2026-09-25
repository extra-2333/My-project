import importlib
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

try:
    from transformers import ViTImageProcessor
except ImportError:
    ViTImageProcessor = None

try:
    from transformers import TFViTModel
except ImportError:
    try:
        vit_modeling = importlib.import_module("transformers.models.vit.modeling_tf_vit")
        TFViTModel = getattr(vit_modeling, "TFViTModel")
    except (ImportError, AttributeError):
        TFViTModel = None

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pensamiento import DeepFullyConnectedNet
else:
    from .pensamiento import DeepFullyConnectedNet

EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp")


class VisionEncoder:
    def __init__(self, vit_name="google/vit-base-patch16-224", freeze_backbone=True):
        if ViTImageProcessor is None or TFViTModel is None:
            raise RuntimeError(
                "La versión instalada de transformers no incluye TFViTModel. "
                "Instala una versión compatible de TensorFlow con Transformers "
                "(pip install tf-keras)."
            )
        self.processor = ViTImageProcessor.from_pretrained(vit_name)
        self.backbone = TFViTModel.from_pretrained(vit_name)
        self.backbone.trainable = not freeze_backbone
        self.feature_dim = self.backbone.config.hidden_size

    @staticmethod
    def read_image_cv2(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def preprocess(self, images_rgb):
        inputs = self.processor(images=images_rgb, return_tensors="tf")
        return inputs["pixel_values"]

    def __call__(self, pixel_values, training=False):
        outputs = self.backbone(pixel_values=pixel_values, training=training)
        cls_token = outputs.last_hidden_state[:, 0, :]  # embedding [CLS]
        return cls_token


class VisionModel(tf.keras.Model):
    """Visión artificial completa: imagen -> características -> decisión."""

    def __init__(self, num_classes, hidden_dim=512, num_hidden_layers=100, **kwargs):
        super().__init__(**kwargs)
        self.encoder = VisionEncoder()
        self.head = DeepFullyConnectedNet(
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_hidden_layers=num_hidden_layers,
        )

    def call(self, pixel_values, training=False):
        features = self.encoder(pixel_values, training=training)
        return self.head(features, training=training)

    def predict_from_path(self, image_path):
        img_rgb = self.encoder.read_image_cv2(image_path)
        pixel_values = self.encoder.preprocess(img_rgb)
        logits = self(pixel_values, training=False)
        return int(tf.argmax(logits, axis=1).numpy()[0])


def listar_dataset_imagenes(dataset_dir):
    """
    Espera esta estructura de carpetas:
        dataset_dir/
            clase_1/ imagen1.jpg imagen2.jpg ...
            clase_2/ imagen1.jpg ...
    Devuelve (rutas, etiquetas, nombres_de_clase).
    """
    clases = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    rutas, etiquetas = [], []
    for idx, clase in enumerate(clases):
        carpeta = os.path.join(dataset_dir, clase)
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(EXTENSIONES_IMAGEN):
                rutas.append(os.path.join(carpeta, archivo))
                etiquetas.append(idx)
    return rutas, etiquetas, clases


def extraer_features_vision(dataset_dir, encoder):
    
    rutas, etiquetas, clases = listar_dataset_imagenes(dataset_dir)
    if not rutas:
        raise ValueError(f"No se encontraron imágenes en: {dataset_dir}")

    features = []
    for i, ruta in enumerate(rutas, start=1):
        img_rgb = VisionEncoder.read_image_cv2(ruta)
        pixel_values = encoder.preprocess(img_rgb)
        emb = encoder(pixel_values, training=False)
        features.append(emb.numpy()[0])
        if i % 50 == 0 or i == len(rutas):
            print(f"  [visión] procesadas {i}/{len(rutas)} imágenes")

    X = np.array(features, dtype="float32")
    y = np.array(etiquetas, dtype="int64")
    return X, y, clases
