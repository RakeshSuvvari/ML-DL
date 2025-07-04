# Code Example (Image Classification using MobileNetV2 on Cats vs Dogs)

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load dataset
(train_ds, val_ds), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Preprocessing function
IMG_SIZE = 160

def format_example(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # Normalize
    return image, label

train = train_ds.map(format_example).batch(32).prefetch(tf.data.AUTOTUNE)
val = val_ds.map(format_example).batch(32).prefetch(tf.data.AUTOTUNE)

# Load MobileNetV2 without top layer
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False  # Freeze base

# Add custom top layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train, validation_data=val, epochs=3)

# Visualize a few predictions
for image, label in val.take(1):
    pred = model.predict(image)
    for i in range(5):
        plt.imshow(image[i])
        plt.title(f"Predicted: {'Dog' if pred[i]>0.5 else 'Cat'}")
        plt.axis('off')
        plt.show()


# Summary
# Feature	            Explanation
# ---------             ----------------
# Pretrained Base	    MobileNetV2 (or ResNet, VGG, etc.)
# Frozen Layers	        Keeps base weights unchanged
# Custom Head	        New layers for your specific task
# Advantage	            Faster training + better accuracy