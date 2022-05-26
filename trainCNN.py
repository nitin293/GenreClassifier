import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import argparse


def load_data(data_path):
    try:
        with open(data_path, "r") as fp:
            data = json.load(fp)

        X = np.array(data["mfcc"])
        y = np.array(data["labels"])

        return X, y

    except:
        print("FAILED LOADING DATA!")
        exit()


def prepare_datasets(DATA_PATH, test_size, validation_size):
    try:
        X, y = load_data(DATA_PATH)

        n_classes = len(set(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

        X_train = X_train[..., np.newaxis]
        X_validation = X_validation[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        return X_train, X_validation, X_test, y_train, y_validation, y_test, n_classes

    except:
        print("ERROR PREPARING DATASET!")
        exit()


def build_model(input_shape, n_classes, LEARNING_RATE):
    try:
        model = keras.Sequential()

        # 1st conv layer
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(keras.layers.BatchNormalization())

        # 2nd conv layer
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(keras.layers.BatchNormalization())

        # 3rd conv layer
        model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        model.add(keras.layers.BatchNormalization())

        # flatten output and feed it into dense layer
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.3))

        # output layer
        model.add(keras.layers.Dense(n_classes, activation='softmax'))

        optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimiser,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    except:
        print("MODEL BUILDING FAILED!")
        exit()


def predict(model, X, y):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


def trainCNN(DATA_PATH, TEST_SIZE, VALIDATION_SIZE, LEARNING_RATE, BATCH_SIZE, EPOCHS, OUTPUT_MODEL):

    X_train, \
    X_validation, \
    X_test, y_train, \
    y_validation, \
    y_test,\
    n_classes = prepare_datasets(DATA_PATH=DATA_PATH, test_size=TEST_SIZE, validation_size=VALIDATION_SIZE)


    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    model = build_model(input_shape, n_classes=n_classes, LEARNING_RATE=LEARNING_RATE)
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=BATCH_SIZE, epochs=EPOCHS)

    if OUTPUT_MODEL:
        model.save(OUTPUT_MODEL)

    for i in range(5):
        # pick a sample to predict from the test set
        X_to_predict = X_test[i]
        y_to_predict = y_test[i]

        # predict sample
        predict(model, X_to_predict, y_to_predict)

    print(input_shape)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        help="JSON Dataset File",
        type=str,
        required=True
    )
    parser.add_argument(
        "--test",
        help="Test Size",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--valid",
        help="Validation Size",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "-b", "--batch",
        help="Batch Size",
        type=int,
        default=32
    )
    parser.add_argument(
        "-e", "--epochs",
        help="Epochs",
        type=int,
        default=40
    )
    parser.add_argument(
        "-lr", "--learningrate",
        help="Learning Rate",
        type=float,
        default=0.00001
    )
    parser.add_argument(
        "-o", "--output",
        help="Output Model",
        type=str,
        default=None
    )
    args = parser.parse_args()

    DATA_PATH = args.dataset
    TEST_SIZE = args.test
    VALIDATION_SIZE = args.valid
    LEARNING_RATE = args.learningrate
    OUTPUT_MODEL = args.output
    BATCH_SIZE = args.batch
    EPOCHS = args.epochs

    trainCNN(
        DATA_PATH=DATA_PATH,
        TEST_SIZE=TEST_SIZE,
        VALIDATION_SIZE=VALIDATION_SIZE,
        LEARNING_RATE=LEARNING_RATE,
        OUTPUT_MODEL=OUTPUT_MODEL,
        BATCH_SIZE=BATCH_SIZE,
        EPOCHS=EPOCHS
    )