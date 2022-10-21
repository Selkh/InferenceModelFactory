# COCO 2017 Dataset Preparation Guide
## Step
1. Download [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset). Put it under a directory.

2. Run commands below

```shell
pip3 install -r requirements.txt
python3 convert_coco2017.py --input_path=<path/to/the/directory/you/containing/coco2017.zip> --output_path=<path/to/data>
```

## Processed Data Structure

```shell
data/
├── annotations
├── test2017
├── train2017
└── val2017
```
