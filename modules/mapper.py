import os
import pandas as pd
import argparse

def mapper(DATA_PATH, OUTPUT_FILE):
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
            dataset.to_csv(f"./{OUTPUT_FILE}", index=False)

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
    parser.add_argument(
        "-o", "--output",
        help="Output CSV filename",
        required=True
    )
    args = parser.parse_args()

    DATA_PATH = args.path
    OUTPUT_FILE = args.output

    mapper = mapper(DATA_PATH, OUTPUT_FILE)

    if mapper:
        print("[+] DONE !")
    else:
        print("[!] FAILED !")