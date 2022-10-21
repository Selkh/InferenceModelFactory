# PASCAL VOC 2012 数据集准备指南

## 步骤

1. 下载 [archive.zip](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012)。把它放在一个文件夹下。

2. 运行下面的命令

```shell
pip3 install -r requirements.txt
python3 convert_voc2012.py --input_path=<你刚创建的文件夹> --output_path=<目标路径>
```

## 处理完的数据结构

```shell
data/
└── VOC2012
```

