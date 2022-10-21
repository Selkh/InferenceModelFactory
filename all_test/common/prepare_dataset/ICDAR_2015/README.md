# ICDAR 2015 Dataset Preparation Guide

## Step

1. Download [ICDAR 2015 test set](https://rrc.cvc.uab.es/?ch=4&com=downloads)(Registration is required for downloading). After registering and logging in, download the "Test Set Images" and "Test Set Ground Truth" in section "Task 4.1: Text Localization (2015 edition)". And, the content downloaded by Test Set Images is saved as the folder ch4_test_images and Test Set Ground Truth in folder ch4_test_localization_transcription_gt.
![ICDAR_2015](./ICDAR_2015.png)
2. Decompress the test set, as follows,

``` shell
cd path/to/ch4_test_images
unzip ch4_test_images.zip
cd path/to/ch4_test_localization_transcription_gt
unzip ch4_test_localization_transcription_gt.zip
```

3. Download the [PaddleOCR format annotation file](https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt). Put it under the same folder of 'ch4_test_images'.
4. Download https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json. Put it under the same folder of 'ch4_test_images'.And run python3 modify_directory.py。

## Processed Dataset Structure

```shell
data/
├── ch4_test_images
│        ├── img_1.jpg
│        ├── img_2.jpg
|        └── ……
├── ch4_test_localization_transcription_gt
│        ├── gt_img_1.txt
│        ├── gt_img_2.txt
|        └── ……
├── instances_test.json
└── test_icdar2015_label.txt
```

