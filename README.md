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

> STEP 1: Install Requirements  
```
$ python3 -m pip install -r requirements.txt
```

> STEP 2: Build the Mapped Dataset
```
$ python3 dataset generator -d <Dataset> -o <dataset>.csv
```

