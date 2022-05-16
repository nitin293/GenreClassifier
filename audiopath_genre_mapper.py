import os
import glob
import pandas as pd

"""# **Mapping {AUDIO -> GENRE} into CSV**"""

def audio_mapper(ORIGIN_DIR):
    try:
        LABELS = os.listdir(ORIGIN_DIR)

        dataframes = []

        for LABEL in LABELS:
            data_dir = f"{ORIGIN_DIR}/{LABEL}"
            files = glob.glob(f"{data_dir}/*.wav")
            data = {"file": files, "genre": [LABEL]*len(files)}

            df = pd.DataFrame(data)
            dataframes.append(df)

            dataset = pd.concat(dataframes, ignore_index=True)
            dataset.to_csv("./file_label.csv", index=False)

        return True

    except:
        return False



if __name__=="__main__":
    ORIGIN_DIR = "/content/drive/MyDrive/Genre-Classification/Data/genres_original"
    audio_mapper(ORIGIN_DIR)



