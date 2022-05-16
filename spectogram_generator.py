import re
import os
import scipy.io.wavfile as wavfile
import pandas as pd
import matplotlib.pyplot as plt
import threading
import argparse


def generate_spectogram(wav_file, outfile):
    rate, data = wavfile.read(wav_file)
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=384, NFFT=512)
    ax.axis('off')
    fig.savefig(outfile, dpi=300, frameon='false')


def runner(csv_filename, threads):
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

            thread = threading.Thread(target=generate_spectogram, args=(filename, output_file, ))
            thread.start()

            # print(filename, output_file)
            # generate_spectogram(filename, output_file)
            print(f"COUNT: {i}", end="\r")

            if i%threads==0:
                thread.join()
            elif i==len(aud_dataset):
                thread.join()

        return True
            
    except:
        exit()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Mapped CSV filename",
        required=True
    )
    parser.add_argument(
        "-t", "--thread",
        type=int,
        help="Threads",
        default=1
    )
    args = parser.parse_args()

    csv_file = args.file
    thread = args.thread

    runner(csv_filename=csv_file, threads=thread)
