from modules import img_mapper
import argparse
import os
import re

def prepare_dataset(AUDIO_DATA_PATH, OUTPUT_CSV, SHAPE=None):
    try:
        img_mapper.generate(data_path=AUDIO_DATA_PATH, output_csv=OUTPUT_CSV, shape=SHAPE)
        os.remove("__temp_map__.csv")
        return True

    except:
        raise


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
    parser.add_argument(
        "-s", "--shape",
        help="Image size/shape tuple",
        default=None,
        type=str
    )
    args = parser.parse_args()

    DATA_PATH = args.data
    OUTPUT_CSV = args.output

    SHAPE = args.shape
    SHAPE = re.findall('[0-9]+', SHAPE)
    SHAPE = tuple(int(size) for size in SHAPE)

    prepare_dataset(AUDIO_DATA_PATH=DATA_PATH, OUTPUT_CSV=OUTPUT_CSV, SHAPE=SHAPE)