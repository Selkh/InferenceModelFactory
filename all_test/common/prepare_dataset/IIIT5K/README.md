# IIIT5K preporcessed Dataset Preparation Guide
## Step
The IIIT 5K-word dataset is harvested from Google image search. Query words like billboards, signboard, house numbers, house name plates, movie posters were used to collect images. The dataset contains 5000 cropped word images from Scene Texts and born-digital images. The dataset is divided into train and test parts. This dataset can be used for large lexicon cropped word recognition. We also provide a lexicon of more than 0.5 million dictionary words with this dataset.

1. download dataset from http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz
2. `tar xf <path to IIIT5K-Word_V3.0.tar.gz>`
3. download label from https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt
4. put test_label.txt into `<IIIT5K folder>`

## Processed Dataset Structure
```
IIIT5K
├── lexicon.txt
├── README
├── test
├── testCharBound.mat
├── testdata.mat
├── test_label.txt
├── train
├── trainCharBound.mat
└── traindata.mat
```