from modules import img_mapper
import argparse
import os

def prepare_dataset(AUDIO_DATA_PATH, OUTPUT_CSV):
    try:
        img_mapper.generate(data_path=AUDIO_DATA_PATH, output_csv=OUTPUT_CSV)
        os.remove("__temp_map__.csv")
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