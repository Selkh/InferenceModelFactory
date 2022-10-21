# DIV2k Dataset Preparation Guide

## Step

1. Download [DIV2K_valid_HR.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip), [DIV2K_valid_LR_bicubic_X4.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip). Put these files under one directory.

2. Run commands below

```shell
pip3 install -r requirements.txt
python3 convert_div2k.py --input_path=<path/to/the/directory/containing/the/two/files> --output_path=<path/to/data>
```

## Processed Dataset Structure

```shell
data/
├── DIV2K_valid_HR
└── DIV2K_valid_LR_bicubic
```
