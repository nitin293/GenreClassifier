import tensorflow as tf
import librosa
import numpy as np
import cv2
import re
import librosa.display
import matplotlib.pyplot as plt
import argparse
from pydub import AudioSegment

def to_wav(src, dst):
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

def reshape(image, shape):
    img = cv2.imread(image)
    img = cv2.resize(img, shape)
    cv2.imwrite(image, img)


def generate_spectogram(wav_file, outfile, shape=None):
    try:
        data, sample_rate = librosa.load(wav_file)
        mel = librosa.feature.melspectrogram(y=data.astype("float64"), sr=sample_rate)
        fig, ax = plt.subplots()
        mel_sgram = librosa.amplitude_to_db(mel, ref=np.min)
        librosa.display.specshow(mel_sgram, sr=sample_rate)
        plt.savefig(outfile)
        plt.close()

        if shape:
            reshape(image=outfile, shape=shape)

    except ValueError:
        print(f"ERROR IN FILE: {wav_file}")
        pass


def load_and_prep_image(filename, img_shape):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size = [img_shape, img_shape])
    img = img/255.

    return img


def predict(audio_file, shape, MODEL):
    genres = sorted([
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
    ])

    filename = '.'.join(audio_file.split('.')[:-1])
    img_filename = f"{filename}.png"

    if audio_file.endswith(".mp3"):
        out_wav = f"{filename}.wav"
        to_wav(src=audio_file, dst=out_wav)

    else:
        out_wav = audio_file

    generate_spectogram(out_wav, img_filename, shape)
    model = tf.keras.models.load_model(MODEL)

    img = load_and_prep_image(img_filename, img_shape=shape[0])
    img = tf.expand_dims(img, axis=0)
    y_prob = model.predict(img)
    y_class = y_prob.argmax(axis=-1)[0]

    return genres[y_class]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--audio",
        help="Audio file",
        type=str,
        required=True
    )
    parser.add_argument(
        "-m", "--model",
        help="Model Directory",
        type=str,
        required=True
    )
    parser.add_argument(
        "-s", "--shape",
        help="Image shape",
        type=str,
        default="128,128"
    )
    args = parser.parse_args()

    AUDIO = args.audio
    SHAPE = args.shape
    MODEL = args.model
    SHAPE = re.findall('[0-9]+', SHAPE)
    SHAPE = tuple(int(size) for size in SHAPE)

    genre = predict(audio_file=AUDIO, shape=SHAPE, MODEL=MODEL)
    print(f"GENRE: {genre}")
