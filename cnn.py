import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.datasets import mnist


## Data import, analysis and preprocessing

# Model params
num_classes = 10
input_shape = (28, 28, 1)

# Import training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# View dimensions of data
print(x_train.shape,y_train.shape)

# Normalise images and labels
x_train = x_train.astype("float32") / 255.0
y_train = y_train.astype("float32") / 255.0

# Reshape images (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape,y_train.shape)

# CNN Architecture Design
model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size = (2,2)))
model.add(layers.Conv2D(64, kernel_size=(3,3), activation="relu"))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(num_classes, activation="softmax"))

# View architecture summary
model.summary()
