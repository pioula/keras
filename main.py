#%%
import argparse

parser = argparse.ArgumentParser(description='Example of command line arguments')
parser.add_argument('--path', type=str, help='an argument')
parser.add_argument('--img', type=int, help='an argument')
parser.add_argument('--ep', type=int, help='an argument')
args = parser.parse_args()

NUMBER_OF_IMAGES = args.img
NUMBER_OF_EPOCHS = args.ep

base_path = args.path
#%% 
import pandas as pd
df = pd.read_csv(f'{base_path}BigDataCup2022/S1/train/train.csv')
df.head()

#%%

# df has columns id, input_path, encoded_path which contains id path of one image and a path to its corresponding encoded version
# Import first 100 images into the numpy array
import numpy as np
import os
import cv2

NUMBER_OF_IMAGES = NUMBER_OF_IMAGES

images = np.zeros((NUMBER_OF_IMAGES * 2, 2, 512, 512, 3), dtype=np.uint8)

# Write a for which will loop over first 100 images and save them into a numpy array
for i in range(NUMBER_OF_IMAGES):
    # Read the image
    img = cv2.imread(f'{base_path}{df.iloc[i,1]}')
    encoded = cv2.imread(f'{base_path}{df.iloc[i,2]}')
    # Save the image into the numpy array
    images[i][0] = img
    images[i][1] = encoded

images.shape
# %% no classes
import random
for i in range(NUMBER_OF_IMAGES):
    img = images[i][0]
    i2 = i
    while i2 == i:
        i2 = random.randint(0, NUMBER_OF_IMAGES - 1)
    enc = images[i2][1]
    images[i + NUMBER_OF_IMAGES][0] = img
    images[i + NUMBER_OF_IMAGES][1] = enc
#%% labels
labels = np.array([1] * NUMBER_OF_IMAGES + [0] * NUMBER_OF_IMAGES)
#%% six channels
X = np.array([cv2.merge((img[0][:,:,0], img[0][:,:,1], img[0][:,:,2], img[1][:,:,0], img[1][:,:,1], img[1][:,:,2])) for img in images], dtype=float)
X /= 255.
#%% Train validation  split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)
# %%
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Define the ResNet152 model
base_model = keras.applications.ResNet152(
    weights=None,  # Load weights pre-trained on ImageNet.
    input_shape=(512, 512, 6),
    include_top=False,
) 
model = tf.keras.Sequential()
model.add(keras.applications.ResNet152(include_top=False, weights=None, input_shape=(512, 512, 6)))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# # Data augmentation
# data_augmentation = tf.keras.Sequential(
#     [
#         layers.RandomFlip("horizontal_and_vertical"),
#         layers.RandomRotation(0.2),
#         layers.RandomZoom(0.2),
#     ]
# )

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
print("fitting")
# Fit the model to the data
model.fit(
    x=X_train,
    y=y_train,
    epochs=NUMBER_OF_EPOCHS,
    validation_data=(X_val, y_val),
)
predictions = model.predict(X_val)
print()
# %%
