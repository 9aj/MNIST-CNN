import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers, models, utils
from keras.datasets import mnist

# Model params
num_classes = 10
input_shape = (28, 28, 1)

# Import training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalise images and labels
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape images (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape,y_train.shape)

# Convert class vectors
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

print("[INFO] Training/testing data loaded")

# Get best model from 15 epochs
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
best = tf.train.latest_checkpoint(checkpoint_dir)

print("[INFO] Loaded checkpoint from previous training")

# Create new basic model (untrained)
model = tf.keras.create_model()

print("[INFO] Creating basic untrained model")

# Load model weights
model.load_weights(best)

print("[INFO] Loaded weights successfully")

loss, acc = model.evaluate(y_train, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))



