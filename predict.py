import numpy as np
import librosa
from pydub import AudioSegment
import re
import argparse
import tensorflow.keras as keras


def to_wav(src, dst):
    print(src, dst)
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

    return True


def load_data(data_dict):
    X = np.array(data_dict["mfcc"])

    return X


def get_mfcc(FILE_PATH, SAMPLE_RATE=22050, num_mfcc=13, n_fft=2048, hop_length=512):
    data = {
        "mfcc": []
    }

    try:
        signal, sample_rate = librosa.load(FILE_PATH, sr=SAMPLE_RATE)

        mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        data["mfcc"].append(mfcc.tolist())

        return data

    except:
        pass


def build_model(input_shape, n_classes):
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



def predict(FILE_PATH, MODEL_PATH, SHAPE):
    labels = [
        'blues',
        'classical',
        'country',
        'disco',
        'hiphop',
        'jazz',
        'metal',
        'pop',
        'reggae',
        'rock'
    ]

    if FILE_PATH.endswith(".mp3"):
        print("CONVERTING TO WAV")
        OUT_PATH = f"{FILE_PATH[:-4]}.wav"
        to_wav(FILE_PATH, OUT_PATH)
        FILE_PATH = OUT_PATH

    model = keras.models.load_model(MODEL_PATH)
    model.summary()

    data = get_mfcc(FILE_PATH)

    X = np.array(data["mfcc"][0])
    X = np.resize(X, SHAPE)
    X = X[np.newaxis, ...]

    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)

    return labels[predicted_index[0]]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--audio',
        help="Audio File with Path",
        type=str,
        required=True
    )
    parser.add_argument(
        '-m', '--model',
        help="Saved Model Directory",
        type=str,
        required=True
    )
    parser.add_argument(
        '-s', '--shape',
        help="Desired shape of input file",
        type=str,
        required=True
    )
    args = parser.parse_args()


    FILE_PATH = args.audio
    MODEL_PATH = args.model
    SHAPE = args.shape
    SHAPE = re.findall('[0-9]+', SHAPE)
    SHAPE = tuple([int(s) for s in SHAPE])

    predicted = predict(FILE_PATH=FILE_PATH, MODEL_PATH=MODEL_PATH, SHAPE=SHAPE)
    print(predicted)