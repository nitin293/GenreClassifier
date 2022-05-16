import re
import os
import numpy as np
import scipy.io.wavfile as wavfile
import pandas as pd
import matplotlib.pyplot as plt
import threading
import argparse
import librosa
import librosa.display


def generate_spectogram(wav_file, outfile):
    # print(f"{wav_file}: {outfile}")
    sample_rate, data = wavfile.read(wav_file)
    mel = librosa.feature.melspectrogram(y=data.astype("float64"), sr=sample_rate)
    fig, ax = plt.subplots()
    mel_sgram = librosa.amplitude_to_db(mel, ref=np.min)
    librosa.display.specshow(mel_sgram, sr=sample_rate)

    plt.savefig(outfile)


def runner(csv_filename):
    try:
        aud_dataset = pd.read_csv(csv_filename)

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

            generate_spectogram(filename, output_file)

            print(f"COUNT: {i}", end="\r")

        return True
            
    except:
        raise


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Mapped CSV filename",
        required=True
    )
    args = parser.parse_args()

    csv_file = args.file

    runner(csv_filename=csv_file)
