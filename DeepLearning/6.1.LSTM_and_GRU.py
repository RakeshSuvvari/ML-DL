import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Load data
# Simple LSTM and GRU for Sentiment Classification (IMDB dataset)
num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences
maxlen = 200
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

def build_model(rnn_type='LSTM'):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=num_words, output_dim=64, input_length=maxlen),
        tf.keras.layers.LSTM(64) if rnn_type == 'LSTM' else tf.keras.layers.GRU(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train LSTM model
print("\n🔁 Training LSTM...")
model_lstm = build_model('LSTM')
history_lstm = model_lstm.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Train GRU model
print("\n🔁 Training GRU...")
model_gru = build_model('GRU')
history_gru = model_gru.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['val_accuracy'], label='LSTM Val Acc')
plt.plot(history_gru.history['val_accuracy'], label='GRU Val Acc')
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_lstm.history['val_loss'], label='LSTM Val Loss')
plt.plot(history_gru.history['val_loss'], label='GRU Val Loss')
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
