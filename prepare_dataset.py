import json
import os
import math
import librosa
import argparse


def save_mfcc(dataset_path, json_path, SAMPLE_RATE, TRACK_DURATION, hop_length, num_segments):

    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                    for d in range(num_segments):
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        mfcc = librosa.feature.mfcc(
                            signal[start:finish],
                            sample_rate,
                            hop_length=hop_length
                        )
                        mfcc = mfcc.T

                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print("GENRE: {} FILE:{}, SEGMENT:{}".format(semantic_label, file_path, d+1), end="\r")

                except:
                    pass
                    continue

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        help="Audio Dataset Path",
        type=str,
        required=True
    )
    parser.add_argument(
        "-o", "--output",
        help="JSON Output File",
        type=str,
        required=True
    )
    parser.add_argument(
        "-sr", "--samplerate",
        help="Sample Rate",
        type=int,
        default=22050
    )
    parser.add_argument(
        "-du", "--duration",
        help="Track Duration",
        type=int,
        default=30
    )
    parser.add_argument(
        "-hl", "--hoplength",
        help="Hop length",
        type=int,
        default=512
    )
    parser.add_argument(
        "-s", "--segments",
        help="No. of Segments",
        type=int,
        default=10
    )
    args = parser.parse_args()


    DATASET_PATH = args.dataset
    JSON_PATH = args.output
    SAMPLE_RATE = args.samplerate
    TRACK_DURATION = args.duration
    HOP_LENGTH = args.hoplength
    NUM_SEGMENTS = args.segments


    save_mfcc(
        dataset_path=DATASET_PATH,
        json_path=JSON_PATH,
        SAMPLE_RATE=SAMPLE_RATE,
        TRACK_DURATION=TRACK_DURATION,
        hop_length=HOP_LENGTH,
        num_segments=10
    )

