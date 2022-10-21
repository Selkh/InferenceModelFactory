# Market 1501 Dataset Preparation Guide

## Step

1. Download the dataset from [Baidu netdisk](https://pan.baidu.com/s/1ntIi2Op) or [Google Drive](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view).

2. Generate dataset with the following command

```shell
python3 convert_market1501.py --dataset <path/to/the/compressed/dataset> --output <path/to/output/directory>
```

## Processed Dataset Structure

```shell
data/
├── query
├── gt_query
├── gt_bbox
├── bounding_box_train
└── bounding_box_test
```

