import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1: Load a small text corpus
text = tf.keras.utils.get_file('shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
with open(text, 'r') as f:
    text = f.read()

print("Sample text:", text[:500])

# Step 2: Create character index mappings
chars = sorted(set(text))
char2idx = {u:i for i, u in enumerate(chars)}
idx2char = np.array(chars)
vocab_size = len(chars)

# Step 3: Encode the text
text_as_int = np.array([char2idx[c] for c in text])

# Step 4: Create input sequences
seq_length = 100
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(seq):
    return seq[:-1], seq[1:]

dataset = sequences.map(split_input_target)
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Step 5: Build the LSTM model
embedding_dim = 256
rnn_units = 512

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True),
    tf.keras.layers.Dense(vocab_size)
])

# Step 6: Loss function
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Step 7: Train the model
history = model.fit(dataset, epochs=20)

# Step 8: Text generation function
def generate_text(model, start_string, num_chars=400, temperature=1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    result = [start_string]

    for _ in range(num_chars):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :] / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        result.append(idx2char[predicted_id])
        input_eval = tf.expand_dims([predicted_id], 0)

    return ''.join(result)


# Step 9: Generate and display text
print("\n Generated Text:\n")
print(generate_text(model, start_string="To be, or not to be", temperature=0.7))
