# Market 1501 数据集准备指南

## 步骤

1. 从[百度网盘](https://pan.baidu.com/s/1ntIi2Op)或者[谷歌云盘](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)下载数据集。

2. 运行下面的命令

```shell
python3 convert_market1501.py --dataset <path/to/the/compressed/dataset> --output <path/to/output/directory>
```

## 处理完成的数据结构

```shell
data/
├── query
├── gt_query
├── gt_bbox
├── bounding_box_train
└── bounding_box_test
```
