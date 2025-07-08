import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class SasokEmotionModel:
    """
    SASOK – Specialized Adaptive System for Optimal Knowledge
    Custom convolutional architecture for facial emotion recognition
    """

    def __init__(self, input_shape: tuple = (48, 48, 1), num_classes: int = 7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model: Model | None = None
        self.history = None
        # Russian emotion labels for FER-like datasets
        self.emotion_labels = [
            "Злость",
            "Отвращение",
            "Страх",
            "Счастье",
            "Грусть",
            "Удивление",
            "Нейтральность",
        ]

    # ---------------------------------------------------------------------
    # Building blocks
    # ---------------------------------------------------------------------
    def _attention_block(self, x, filters: int):
        """Combined spatial and channel attention block."""
        # Spatial attention (global)
        spatial = layers.GlobalAveragePooling2D()(x)
        spatial = layers.Dense(filters // 8, activation="relu")(spatial)
        spatial = layers.Dense(filters, activation="sigmoid")(spatial)
        spatial = layers.Reshape((1, 1, filters))(spatial)

        # Channel attention (spatial conv)
        channel = layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")(x)

        # Merge attentions
        x = layers.Multiply()([x, spatial])
        x = layers.Multiply()([x, channel])
        return x

    def _residual_block(self, x, filters: int, kernel_size: int = 3):
        """Residual CNN block with optional projection shortcut."""
        shortcut = x

        # First conv
        x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Second conv
        x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        # Match channels if necessary
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Residual add & activation
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        return x

    # ---------------------------------------------------------------------
    # Model assembly & compilation
    # ---------------------------------------------------------------------
    def _build_model(self) -> Model:
        inputs = layers.Input(shape=self.input_shape)

        # Initial conv
        x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Block 1
        x = self._residual_block(x, 64)
        x = self._attention_block(x, 64)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)

        # Block 2
        x = self._residual_block(x, 128)
        x = self._attention_block(x, 128)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)

        # Block 3
        x = self._residual_block(x, 256)
        x = self._attention_block(x, 256)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.3)(x)

        # Block 4
        x = self._residual_block(x, 512)
        x = self._attention_block(x, 512)
        x = layers.GlobalAveragePooling2D()(x)

        # Multi-branch dense classifier
        branch1 = layers.Dense(256, activation="relu")(x)
        branch1 = layers.BatchNormalization()(branch1)
        branch1 = layers.Dropout(0.5)(branch1)
        branch1 = layers.Dense(128, activation="relu")(branch1)

        branch2 = layers.Dense(128, activation="relu")(x)
        branch2 = layers.BatchNormalization()(branch2)
        branch2 = layers.Dropout(0.4)(branch2)
        branch2 = layers.Dense(64, activation="relu")(branch2)

        combined = layers.Concatenate()([branch1, branch2])
        combined = layers.Dense(128, activation="relu")(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.5)(combined)

        outputs = layers.Dense(self.num_classes, activation="softmax", name="emotion_output")(combined)

        return Model(inputs, outputs, name="SASOK_Emotion_Model")

    def compile(self, learning_rate: float = 0.001) -> Model:
        """Build and compile the model if not already done."""
        if self.model is None:
            self.model = self._build_model()

        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )

        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )
        return self.model

    # ---------------------------------------------------------------------
    # Data utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _data_generators(x_train, y_train, x_val, y_val, batch_size: int = 32):
        """Create ImageDataGenerator instances for training/validation."""
        train_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,
            shear_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
            rescale=1.0 / 255,
        ).flow(x_train, y_train, batch_size=batch_size)

        val_gen = ImageDataGenerator(rescale=1.0 / 255).flow(
            x_val, y_val, batch_size=batch_size
        )
        return train_gen, val_gen

    @staticmethod
    def _callbacks(model_name: str = "sasok_best_model.h5"):
        return [
            ModelCheckpoint(
                model_name,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=7,
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.95 ** epoch),
        ]

    # ---------------------------------------------------------------------
    # Training / evaluation / inference
    # ---------------------------------------------------------------------
    def fit(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        *,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
    ):
        if self.model is None:
            raise ValueError("Call compile() before fit().")

        print("🚀 Starting SASOK training…")
        print(f"📊 Training samples: {x_train.shape[0]}")
        print(f"📊 Validation samples: {x_val.shape[0]}")

        train_gen, val_gen = self._data_generators(
            x_train, y_train, x_val, y_val, batch_size
        )

        start = datetime.now()
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=len(x_train) // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=len(x_val) // batch_size,
            callbacks=self._callbacks(),
            verbose=verbose,
        )
        print(f"⏱ Training time: {datetime.now() - start}")
        return self.history

    def evaluate(self, x_test, y_test):
        if self.model is None:
            raise ValueError("Compile the model first.")
        x_test = x_test.astype("float32") / 255.0
        loss, acc, top_k = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"🎯 Test accuracy: {acc:.4f}")
        print(f"🎯 Top-K accuracy: {top_k:.4f}")
        print(f"📉 Test loss: {loss:.4f}")
        return acc, loss

    def predict_emotion(self, image: np.ndarray):
        if self.model is None:
            raise ValueError("Model not compiled.")

        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        image = image.astype("float32") / 255.0
        preds = self.model.predict(image, verbose=0)
        idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds, axis=1)[0])
        return self.emotion_labels[idx], confidence, preds[0]

    # ---------------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------------
    def plot_history(self):
        if self.history is None:
            print("❌ No training history available.")
            return
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history["accuracy"], label="train")
        plt.plot(self.history.history["val_accuracy"], label="val")
        plt.title("Accuracy")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history["loss"], label="train")
        plt.plot(self.history.history["val_loss"], label="val")
        plt.title("Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def summary(self):
        if self.model is None:
            raise ValueError("Model not built.")
        return self.model.summary()


# -------------------------------------------------------------------------
# Demonstration
# -------------------------------------------------------------------------

def _demo():
    """Quick synthetic data demo for sanity check."""
    print("🎭 SASOK emotion model demo")
    # Synthetic dataset
    x_train = np.random.rand(5000, 48, 48, 1).astype("float32")
    y_train = np.random.randint(0, 7, 5000)
    x_test = np.random.rand(1000, 48, 48, 1).astype("float32")
    y_test = np.random.randint(0, 7, 1000)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    model = SasokEmotionModel()
    model.compile()
    model.summary()
    model.fit(x_train, y_train, x_val, y_val, epochs=5, batch_size=32)
    model.evaluate(x_test, y_test)

    sample = x_test[0]
    emotion, conf, probs = model.predict_emotion(sample)
    print("🔮 Prediction:")
    print(f"Emotion: {emotion} | Confidence: {conf:.3f}")
    for label, p in zip(model.emotion_labels, probs):
        print(f"{label}: {p:.3f}")


if __name__ == "__main__":
    _demo()