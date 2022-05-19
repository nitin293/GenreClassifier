import re
import os
import numpy as np
import scipy.io.wavfile as wavfile
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import librosa
import librosa.display
from modules import mapper
import cv2

def reshape(image, shape):
    img = cv2.imread(image)
    img = cv2.resize(img, shape)
    cv2.imwrite(image, img)


def generate_spectogram(wav_file, outfile, shape=None):
    try:
        sample_rate, data = wavfile.read(wav_file)
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


def csv_map(DATA_PATH, OUTPUT_CSV):
    csv = mapper.mapper(DATA_PATH=DATA_PATH, OUTPUT_FILE=OUTPUT_CSV)

    if csv:
        return True

    else:
        return False


def generate(data_path, output_csv, shape=None):
    try:
        csv_map(DATA_PATH=data_path, OUTPUT_CSV="__temp_map__.csv")
        aud_dataset = pd.read_csv("__temp_map__.csv")

        dir = "spectograms"
        if dir not in os.listdir():
            os.mkdir(dir)

        genres = aud_dataset["genre"]
        for genre in genres:
            genre_dir = f"{dir}/{genre}"
            if genre not in os.listdir(dir):
                os.mkdir(genre_dir)

        for i in range(len(aud_dataset)):
            filename = aud_dataset.iloc[i]["file"]
            genre = aud_dataset.iloc[i]["genre"]

            img_filename = re.findall(f'[a-zA-Z0-9.]+', filename)[-1]
            img_filename = f"{img_filename[:-4]}.png"
            output_file = f"spectograms/{genre}/{img_filename}"

            generate_spectogram(filename, output_file, shape)
            mapper.mapper(DATA_PATH="spectograms", OUTPUT_FILE=output_csv)

            print(f"COUNT: {i}", end="\r")

    except:
        raise


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output CSV filename",
        required=True
    )
    parser.add_argument(
        "-d", "--data",
        type=str,
        help="Audio data path",
        required=True
    )
    args = parser.parse_args()

    output_csv = args.output
    data_path = args.data

    generate(output_csv=output_csv, data_path=data_path)
