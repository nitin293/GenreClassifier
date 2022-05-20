# GenreDetector

A project on detection of Genre of music using Neural Network.


## Dataset Structure
```
<Dataset>
    |
    |--> <Genre 1>
    |        |--> <G1 Audio 1>
    |        |--> <G1 Audio 2>
    |
    |--> <Genre 2>
    |        |--> <G2 Audio 1>
    |        |--> <G2 Audio 2>
    |
    |-> <Genre 3>
             |--> <G3 Audio 1>
             |--> <G3 Audio 2>

```


## Usage

#### **PREPARE DATASET**

> STEP 1: Install Requirements  
```
$ python3 -m pip install -r requirements.txt
```

> STEP 2: Build the Mapped Dataset
```
$ python3 imgDataset.py -d <Dataset> -s <image_width>,<image_height> -o <dataset>.csv
```

> STEP 3: Train Dataset
```
$ python3 trainCNN.py -d <Dataset> -b <batch-size> -e <epochs> -s <image_width>,<image_height> -o <Output-Model-Name>
```

> STEP 4: Test on new Audio file
```
$ python3 predict.py -a <Audio File>
```