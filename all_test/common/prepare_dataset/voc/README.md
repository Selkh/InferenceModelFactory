# PASCAL VOC 2012 Preparation Guide

## Step

1. Download [archive.zip](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012). Put it under a directory.

2. Run commands below

```shell
pip3 install -r requirements.txt
python3 convert_voc2012.py --input_path=<path/to/the/directory/containing/the/file> --output_path=<path/to/data>
```

## Processed Dataset Structure

```shell
data/
└── VOC2012
```
