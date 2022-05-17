import cv2
from modules import img_mapper
import pandas as pd
import argparse


def readIMG(image):
    img = cv2.imread(image)

    return img


def prepare_dataset(AUDIO_DATA_PATH, OUTPUT_CSV):
    try:
        img_mapper.generate(data_path=AUDIO_DATA_PATH, output_csv="__tmp_img_dataset__.csv")
        dataset = pd.read_csv("__tmp_img_dataset__.csv")

        images = []
        genres = []

        for index in range(len(dataset)):
            img_file = dataset.iloc[index]["file"]
            img = readIMG(img_file)
            img_genre = dataset.iloc[index]["genre"]

            images.append(img)
            genres.append(img_genre)

        data = {
            "image": images,
            "genre": genres
        }

        dataframe = pd.DataFrame(data)
        dataframe.to_csv(OUTPUT_CSV, index=False)

        return True

    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data",
        help="Audio Data Path",
        required=True,
        type=str
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV filename",
        required=True,
        type=str
    )
    args = parser.parse_args()

    DATA_PATH = args.data
    OUTPUT_CSV = args.output

    prepare_dataset(AUDIO_DATA_PATH=DATA_PATH, OUTPUT_CSV=OUTPUT_CSV)