import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DeepFullyConnectedNet(keras.Model):
   

    def __init__(self, hidden_dim, output_dim,
                 num_hidden_layers=100, dropout=0.1, use_residual=True, **kwargs):
        super().__init__(**kwargs)
        self.use_residual = use_residual

       
        self.input_block = keras.Sequential([
            layers.Dense(hidden_dim, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Activation("gelu"),
            layers.Dropout(dropout),
        ])

       
        self.hidden_blocks = [
            keras.Sequential([
                layers.Dense(hidden_dim, kernel_initializer="he_normal"),
                layers.BatchNormalization(),
                layers.Activation("gelu"),
                layers.Dropout(dropout),
            ])
            for _ in range(num_hidden_layers)
        ]

        
        self.output_layer = layers.Dense(output_dim)

    def call(self, x, training=False):
        x = self.input_block(x, training=training)
        for block in self.hidden_blocks:
            out = block(x, training=training)
            
            x = x + out if self.use_residual else out
        return self.output_layer(x)


def build_dataset(X, y, batch_size=32, shuffle=True):
    
    ds = tf.data.Dataset.from_tensor_slices((X.astype("float32"), y.astype("int64")))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    return ds.batch(batch_size)


def train_model(model, dataset, epochs=20, lr=1e-3):
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric = keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(1, epochs + 1):
        acc_metric.reset_state()
        running_loss, total = 0.0, 0

        for xb, yb in dataset:
            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                loss = loss_fn(yb, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            acc_metric.update_state(yb, logits)
            running_loss += float(loss) * xb.shape[0]
            total += xb.shape[0]

        print(f"Época {epoch:3d}/{epochs} | "
              f"Pérdida: {running_loss/total:.4f} | "
              f"Precisión: {acc_metric.result().numpy():.4f}")

    return model