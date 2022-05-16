import re
import pandas as pd
import numpy as np
import scipy
import scipy.io.wavfile as wavfile
import librosa
import matplotlib.pyplot as plt


def generate_spectogram(csv_filename): 
  try:
    aud_dataset = pd.read_csv(csv_filename)

    
    for i in range(len(aud_dataset)):  
      filename = aud_dataset.iloc[i]["file"]
      genre = aud_dataset.iloc[i]["genre"]

      img_filename = re.findall(f'[a-zA-Z0-9.]+', filename)[-1]
      img_filename = f"{img_filename[:-3]}png"
      output_path = f"./spectograms/{genre}/{img_filename}"

      samplerate, data = wavfile.read(filename)
      powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(data, Fs=samplerate)
    
      plt.savefig(output_path, frameon='false')

    return True
    
  except:
    return False


if __name__=="__main__":
    csv_file = "./file_label.csv"
    generate_spectogram(csv_file)