"""
Architectures du modèle de classification des stades du sommeil.

- build_cnn_lstm_model : version originale (CNN + Bi-LSTM) — précise mais NON portable sur NPU
                        car le NPU 3720 (Meteor Lake) ne supporte pas les ops Loop / ReverseSequence.
- build_cnn_npu_model  : version CNN pure (1D ConvNet) — entièrement NPU-compatible.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D,
)


def build_cnn_lstm_model(input_shape=(3000, 1), num_classes=5):
    """Modèle CNN + Bi-LSTM — meilleur F1 mais incompatible NPU."""
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),

        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.4),

        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_cnn_npu_model(input_shape=(3000, 1), num_classes=5):
    """
    Modèle CNN 1D pur — compatible Intel NPU AI Boost (3720).
    N'utilise que des ops supportées : Conv1D, BN, ReLU, MaxPool,
    GlobalAveragePooling, Dense, Softmax.
    """
    model = Sequential([
        # Bloc 1 : extraction grossière
        Conv1D(64, kernel_size=11, strides=1, activation='relu',
               padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),       # 3000 → 750
        Dropout(0.2),

        # Bloc 2 : motifs intermédiaires
        Conv1D(128, kernel_size=7, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),       # 750 → 187
        Dropout(0.3),

        # Bloc 3 : motifs fins
        Conv1D(256, kernel_size=5, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),       # 187 → 46
        Dropout(0.3),

        # Bloc 4 : compression
        Conv1D(256, kernel_size=3, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),       # 46 → 23
        Dropout(0.3),

        # Tête de classification
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    print("=== CNN + Bi-LSTM ===")
    build_cnn_lstm_model().summary()
    print("\n=== CNN-only (NPU) ===")
    build_cnn_npu_model().summary()
