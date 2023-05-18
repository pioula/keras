#%%
NUMBER_OF_IMAGES = 10000
BATCH_SIZE = 10
base_path = './X/'
#%%
import argparse

parser = argparse.ArgumentParser(description='Example of command line arguments')
parser.add_argument('--path', type=str, help='an argument')
parser.add_argument('--ep', type=int, help='an argument')
args = parser.parse_args()

NUMBER_OF_EPOCHS = args.ep
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
base_path = args.path


#%%
import random
def cycle_shift(ds):
    dataset0 = ds.enumerate()
    dataset1 = dataset0.filter(lambda x, _: x == 0)
    dataset2 = dataset0.filter(lambda x, _: x != 0)
    dataset3 = dataset2.concatenate(dataset1)
    dataset4 = dataset3.map(lambda _, y: y, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset4

def generator():
    random.seed(42)
    for _ in range(2000):
        yield random.choice([True, False])


def train_val_split(ds, dataset_size, val_fraction):

    # Calculate the number of elements in the validation set
    val_size = int(dataset_size * val_fraction)

    # Split the dataset into training and validation sets
    train_dataset = ds.skip(val_size)
    val_dataset = ds.take(val_size)

    return train_dataset, val_dataset

#%%
import tensorflow as tf
from tensorflow import keras

input = keras.utils.image_dataset_from_directory(
    directory=f'{base_path}BigDataCup2022/S1/train',
    labels="inferred",
    image_size=(512,512),
    shuffle=False,
    batch_size=BATCH_SIZE
    )
encoded = input.skip(NUMBER_OF_IMAGES//BATCH_SIZE) # 1 shape = ((1000, 10, 512, 512, 3), (1000, 10, ))
input = input.take(NUMBER_OF_IMAGES//BATCH_SIZE) # 0 (1000, 10, 512, 512, 3)
print("loaded")
#%%
zipped = tf.data.Dataset.zip((input, encoded))
good_labels = zipped.map(lambda x, y: ((tf.concat((x[0], y[0]), axis=3)), x[1]), num_parallel_calls=tf.data.AUTOTUNE)

encoded_shift = cycle_shift(encoded)
zipped_bad = tf.data.Dataset.zip((input, encoded_shift))
bad_labels = zipped_bad.map(lambda x, y: ((tf.concat((x[0], y[0]), axis=3)), y[1]), num_parallel_calls=tf.data.AUTOTUNE)

good_bad_zipped = tf.data.Dataset.zip((good_labels, bad_labels, tf.data.Dataset.from_generator(generator, output_types=tf.bool)))
shuffle_part1 = good_bad_zipped.map(lambda x, y, z: x if z else y)
shuffle_part2 = good_bad_zipped.map(lambda x, y, z: x if not z else y)

X = shuffle_part1.concatenate(shuffle_part2).map(lambda x, y: (x / 255., y), num_parallel_calls=tf.data.AUTOTUNE)
X = X.prefetch(buffer_size=tf.data.AUTOTUNE)
X_train, X_val = train_val_split(X, (NUMBER_OF_IMAGES//BATCH_SIZE) * 2, 0.2)
#%%
import tensorflow as tf
from tensorflow import keras
from keras import layers

with tf.device('/device:GPU:0'):
    model = tf.keras.Sequential()
    model.add(keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(512, 512, 6)))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

    print("fitting")
    model.fit(X_train, epochs=NUMBER_OF_EPOCHS, batch_size=2, steps_per_epoch=200)
    print("predicting")
    print(model.evaluate(X_val))

# %%

    # # Data augmentation
    # data_augmentation = tf.keras.Sequential(
    #     [
    #         layers.RandomFlip("horizontal_and_vertical"),
    #         layers.RandomRotation(0.2),
    #         layers.RandomZoom(0.2),
    #     ]
    # )