# MOT-16 data Preparation Guide

## Step

1. MOT-16 dataset is from this website: https://motchallenge.net/data/MOT16/, the download link is https://motchallenge.net/data/MOT16.zip.

2. After downloading the dataset, unzip the zip file.


## Processed Dataset Structure

```
MOT16/
├── test
│   ├── MOT16-01
│   │   ├── det
│   │   │   └── det.txt
│   │   ├── img1
│   │   │   ├── 000001.jpg
│   │   │   ├── xxxxxx.jpg
│   │   │   └── 000450.jpg
│   │   └── seqinfo.ini
│   ├── MOT16-03
│   ├── MOT16-06
│   ├── MOT16-07
│   ├── MOT16-08
│   ├── MOT16-12
│   └── MOT16-14
│
└── train
    ├── MOT16-02
    │   ├── det
    │   │   └── det.txt
    │   ├── gt
    │   │   └── gt.txt
    │   ├── img1
    │   │   ├── 000001.jpg
    │   │   ├── xxxxxx.jpg
    │   │   └── 000600.jpg
    │   └── seqinfo.ini
    ├── MOT16-04
    ├── MOT16-05
    ├── MOT16-09
    ├── MOT16-10
    ├── MOT16-11
    └── MOT16-13
```
