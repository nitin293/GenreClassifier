import os
import pandas as pd
import argparse

def audio_mapper(DATA_PATH):
    try:
        LABELS = os.listdir(DATA_PATH)

        dataframes = []

        for LABEL in LABELS:
            data_dir = f"{DATA_PATH}/{LABEL}"
            files = os.listdir(f"{data_dir}")
            files = [f"{data_dir}/{file}" for file in files]
            data = {"file": files, "genre": [LABEL]*len(files)}

            df = pd.DataFrame(data)
            dataframes.append(df)

            dataset = pd.concat(dataframes, ignore_index=True)
            dataset.to_csv("./file_label.csv", index=False)

        return True

    except:
        return False



if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path",
        help="Dataset Path",
        required=True
    )
    args = parser.parse_args()

    DATA_PATH = args.path
    mapper = audio_mapper(DATA_PATH)

    if mapper:
        print("[+] DONE !")
    else:
        print("[!] FAILED !")