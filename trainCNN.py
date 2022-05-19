from modules import dataloader
import tensorflow as tf
from tensorflow.keras import layers, models
import re
import argparse


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def load_data(DIR, BATCH, SHAPE):
    dataset = dataloader.load_data_as_tensor(DIR=DIR, BATCH=BATCH, SHAPE=SHAPE)

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
    n_classes = len(dataset.class_names)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds, val_ds, n_classes


def CNN(n_classes, image_size, batch_size, channels=3):

    IN_SHAPE = (batch_size, image_size[0], image_size[1], channels)

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(SHAPE[0], SHAPE[1]),
        layers.experimental.preprocessing.Rescaling(1./255),
    ])

    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(batch_size, kernel_size=(3, 3), activation='relu', input_shape=IN_SHAPE),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,  kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128,  kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])

    model.build(input_shape=IN_SHAPE)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train(DIR, BATCH, SHAPE, EPOCHS):
    train_ds, test_ds, val_ds, n_classes = load_data(DIR=DIR, BATCH=BATCH, SHAPE=SHAPE)
    model = CNN(n_classes=n_classes, image_size=SHAPE, batch_size=BATCH)

    history = model.fit(
        train_ds,
        batch_size=BATCH,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    return history




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        help="Image dataset root directory path",
        required=True,
        type=str
    )
    parser.add_argument(
        "-b", "--batch",
        help="Batch size",
        default=32,
        type=int
    )
    parser.add_argument(
        "-s", "--shape",
        help="Image size/shape",
        required=True,
        type=str
    )
    parser.add_argument(
        "-e", "--epochs",
        help="Number of epochs",
        default=50,
        type=int,
        required=True
    )
    args = parser.parse_args()

    DIR = args.dataset
    BATCH = args.batch
    EPOCHS = args.epochs

    SHAPE = args.shape
    SHAPE = re.findall('[0-9]+', SHAPE)
    SHAPE = tuple(int(size) for size in SHAPE)

    train(DIR=DIR, BATCH=BATCH, SHAPE=SHAPE, EPOCHS=EPOCHS)
