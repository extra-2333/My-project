import importlib.util
import importlib
import os
import sys

import numpy as np
import tensorflow as tf

try:
    from transformers import Wav2Vec2Processor
except ImportError:
    Wav2Vec2Processor = None

try:
    from transformers import TFWav2Vec2Model
except ImportError:
    try:
        wav2vec_modeling = importlib.import_module("transformers.models.wav2vec2.modeling_tf_wav2vec2")
        TFWav2Vec2Model = getattr(wav2vec_modeling, "TFWav2Vec2Model")
    except (ImportError, AttributeError):
        TFWav2Vec2Model = None

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pensamiento import DeepFullyConnectedNet
else:
    from .pensamiento import DeepFullyConnectedNet


def _require_optional_tts_dependencies():
    missing = []
    if importlib.util.find_spec("soundfile") is None:
        missing.append("soundfile")
    if importlib.util.find_spec("tensorflow_tts") is None:
        missing.append("tensorflow-tts")

    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "Faltan dependencias opcionales para síntesis de voz: "
            f"{packages}. Instálalas con: "
            "\\.venv\\Scripts\\python.exe -m pip install soundfile "
            "&& .\\.venv\\Scripts\\python.exe -m pip install "
            "git+https://github.com/TensorSpeech/TensorFlowTTS.git"
        )


class AudioEncoder:
    def __init__(self, wav2vec_name="facebook/wav2vec2-base-960h", freeze_backbone=True):
        if Wav2Vec2Processor is None or TFWav2Vec2Model is None:
            raise RuntimeError(
                "La versión instalada de transformers no incluye TFWav2Vec2Model. "
                "Instala una versión compatible de TensorFlow con Transformers "
                "(pip install tf-keras)."
            )
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_name)
        self.backbone = TFWav2Vec2Model.from_pretrained(wav2vec_name)
        self.backbone.trainable = not freeze_backbone
        self.feature_dim = self.backbone.config.hidden_size

    @staticmethod
    def load_audio(path, target_sr=16000):
        
        audio_binary = tf.io.read_file(path)
        waveform, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
        waveform = tf.squeeze(waveform, axis=-1)
        return waveform.numpy(), int(sr.numpy())

    def preprocess(self, waveform, sr=16000):
        inputs = self.processor(waveform, sampling_rate=sr, return_tensors="tf")
        return inputs.input_values

    def __call__(self, input_values, training=False):
        outputs = self.backbone(input_values=input_values, training=training)
        embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1)  # pooling temporal
        return embedding


class AudioModel(tf.keras.Model):
    

    def __init__(self, num_classes, hidden_dim=512, num_hidden_layers=100, **kwargs):
        super().__init__(**kwargs)
        self.encoder = AudioEncoder()
        self.head = DeepFullyConnectedNet(
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_hidden_layers=num_hidden_layers,
        )

    def call(self, input_values, training=False):
        features = self.encoder(input_values, training=training)
        return self.head(features, training=training)

    def predict_from_path(self, audio_path):
        waveform, sr = self.encoder.load_audio(audio_path)
        input_values = self.encoder.preprocess(waveform, sr)
        logits = self(input_values, training=False)
        return int(tf.argmax(logits, axis=1).numpy()[0])


class TextToSpeechSynthesizer:
    

    def __init__(self,
                 tacotron2_name="tensorspeech/tts-tacotron2-ljspeech-en",
                 vocoder_name="tensorspeech/tts-mb_melgan-ljspeech-en"):
        _require_optional_tts_dependencies()
        tensorflow_tts_inference = importlib.import_module("tensorflow_tts.inference")
        AutoProcessor = tensorflow_tts_inference.AutoProcessor
        TFAutoModel = tensorflow_tts_inference.TFAutoModel
        self.processor = AutoProcessor.from_pretrained(tacotron2_name)
        self.tacotron2 = TFAutoModel.from_pretrained(tacotron2_name)
        self.vocoder = TFAutoModel.from_pretrained(vocoder_name)

    def synthesize(self, text, output_path="voz_generada.wav", sample_rate=22050):
        soundfile = importlib.import_module("soundfile")

        input_ids = self.processor.text_to_sequence(text)
        _, mel_outputs, _, _ = self.tacotron2.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], dtype=tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32),
        )
        audio = self.vocoder.inference(mel_outputs)[0, :, 0]
        soundfile.write(output_path, audio.numpy(), sample_rate)
        return output_path


def listar_dataset_audio(dataset_dir):
   
    clases = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    rutas, etiquetas = [], []
    for idx, clase in enumerate(clases):
        carpeta = os.path.join(dataset_dir, clase)
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(".wav"):
                rutas.append(os.path.join(carpeta, archivo))
                etiquetas.append(idx)
    return rutas, etiquetas, clases


def extraer_features_audio(dataset_dir, encoder):
    
    rutas, etiquetas, clases = listar_dataset_audio(dataset_dir)
    if not rutas:
        raise ValueError(f"No se encontraron audios .wav en: {dataset_dir}")

    features = []
    for i, ruta in enumerate(rutas, start=1):
        waveform, sr = AudioEncoder.load_audio(ruta)
        input_values = encoder.preprocess(waveform, sr)
        emb = encoder(input_values, training=False)
        features.append(emb.numpy()[0])
        if i % 50 == 0 or i == len(rutas):
            print(f"  [voz] procesados {i}/{len(rutas)} audios")

    X = np.array(features, dtype="float32")
    y = np.array(etiquetas, dtype="int64")
    return X, y, clases
