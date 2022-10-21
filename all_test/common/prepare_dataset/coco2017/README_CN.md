# COCO 2017 数据集准备指南
## 步骤
1. 下载 [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)。把它放到一个文件夹下面。

2. 运行下面的命令

```shell
pip3 install -r requirements.txt
python3 convert_coco2017.py --input_path=<你刚创建的文件夹> --output_path=<目标路径>
```

## 处理完成的数据结构

```shell
data/
├── annotations
├── test2017
├── train2017
└── val2017
```

