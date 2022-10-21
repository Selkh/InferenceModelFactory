# DIV2k 数据集准备指南

## 步骤

1. 下载 [DIV2K_valid_HR.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip), [DIV2K_valid_LR_bicubic_X4.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip)。把它们放到一个文件夹下。

2. 运行下面的命令

```shell
pip3 install -r requirements.txt
python3 convert_div2k.py --input_path=<你刚创建的文件夹> --output_path=<目标路径>
```

## 处理完成的数据结构

```shell
data/
├── DIV2K_valid_HR
└── DIV2K_valid_LR_bicubic
```

