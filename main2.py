# tensorflow_cnn.py
import tensorflow as tf
from keras import layers, models, Input
from keras.datasets import fashion_mnist

# Učitavanje i priprema podataka
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalizacija i reshaping
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Deljenje na trening i validaciju
x_val = x_train[48000:]
y_val = y_train[48000:]
x_train = x_train[:48000]
y_train = y_train[:48000]

x_train = x_train[..., tf.newaxis]
x_val = x_val[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Model CNN
model = models.Sequential([
    Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Kompajliranje modela
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treniranje
model.fit(x_train, y_train, epochs=5, batch_size=64,
          validation_data=(x_val, y_val))

# Evaluacija
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
