from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, Flatten, \
    BatchNormalization


def build_cnn_lstm_model(input_shape=(1, 3000), num_classes=5):
    model = Sequential([
        # --- BLOC 1 : CNN (Extraction de motifs morphologiques) ---
        # Note: input_shape est (canaux, points), on transpose souvent pour Keras
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(3000, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # --- BLOC 2 : Bi-LSTM (Dépendances temporelles) ---
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.4),

        # --- BLOC 3 : Classification finale ---
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')  # 5 classes (W, N1, N2, N3, REM)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    model = build_cnn_lstm_model()
    model.summary()