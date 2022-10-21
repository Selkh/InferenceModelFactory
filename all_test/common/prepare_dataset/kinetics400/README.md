# Kinetics400 Dataset Preparation Guide

## Step

1. Download Kinetics 400 Validation Video compressed file [kinetics_400_val_320.tar](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB).

2. Download Kinetcis 400 label file [kinetics_val_list.txt](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt)

3. Unzip kinetics_400_val_320.tar and rename directory kinetics_400_val_320 to val_320

4. Organize the aforementioned files as the structure shown below.

## Processed Dataset Structure

```shell
data/
├── label
│     └── kinetics_val_list.txt
└── val_320
```

