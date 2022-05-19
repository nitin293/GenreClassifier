import tensorflow as tf

def load_data_as_tensor(DIR, BATCH, SHAPE, SHUFFLE=True):

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DIR,
        shuffle=SHUFFLE,
        image_size=SHAPE,
        batch_size=BATCH
    )

    return dataset