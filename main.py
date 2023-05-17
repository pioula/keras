#%%
NUMBER_OF_IMAGES = 10000
base_path = './'
#%%
import argparse

parser = argparse.ArgumentParser(description='Example of command line arguments')
parser.add_argument('--path', type=str, help='an argument')
parser.add_argument('--ep', type=int, help='an argument')
args = parser.parse_args()

NUMBER_OF_IMAGES = args.img
NUMBER_OF_EPOCHS = args.ep

base_path = args.path
#%%
def cycle_shift(ds):
    dataset0 = ds.enumerate()
    dataset1 = dataset0.filter(lambda x, _: x == 0)
    dataset2 = dataset0.filter(lambda x, _: x != 0)
    dataset3 = dataset2.concatenate(dataset1)
    dataset4 = dataset3.map(lambda _, y: y)
    return dataset4
#%%
import tensorflow as tf
from tensorflow import keras

input = keras.utils.image_dataset_from_directory(
    directory=f'{base_path}BigDataCup2022/S1/train/input',
    labels=None,
    label_mode=None,
    image_size=(512,512),
    shuffle=False
    )
#%%
encoded = keras.utils.image_dataset_from_directory(
    directory=f'{base_path}BigDataCup2022/S1/train/enc',
    labels=None,
    label_mode=None,
    image_size=(512,512),
    shuffle=False
)
#%%
zipped = tf.data.Dataset.zip((input, encoded))
good_labels = zipped.map(lambda x, y: (tf.concat((x, y), axis=3), [1]))

encoded_shift = cycle_shift(encoded)
zipped_bad = tf.data.Dataset.zip((input, encoded_shift))
bad_labels = zipped_bad.map(lambda x, y: (tf.concat((x, y), axis=3), [0]))
X = good_labels.concatenate(bad_labels).map(lambda x, y: (x / 255., y))
#%%
def add_labels(image, label):
    # Your custom label logic here
    return image, tf.expand_dims(label, axis=-1)

dataset = X

new_dataset = dataset.map(lambda x, y: add_labels(x, y))
new_dataset
#%%
import tensorflow as tf
from tensorflow import keras
from keras import layers

with tf.device('/device:GPU:0'):
    model = tf.keras.Sequential()
    model.add(keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(512, 512, 6)))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    print("fitting")
    # Fit the model to the data

    model.fit(X, epochs=5)

    # model.fit(
    #     X, epochs=10
    # )

    # predictions = model.predict(X_val)
    # print(predictions)

    # # round predictions
    # predictions = np.round(predictions)

    # accuracy = tf.keras.metrics.Accuracy()
    # print(accuracy(predictions, y_val))
# %%

    # # Data augmentation
    # data_augmentation = tf.keras.Sequential(
    #     [
    #         layers.RandomFlip("horizontal_and_vertical"),
    #         layers.RandomRotation(0.2),
    #         layers.RandomZoom(0.2),
    #     ]
    # )