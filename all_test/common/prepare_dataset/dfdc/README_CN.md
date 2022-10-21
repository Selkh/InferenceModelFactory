# Deep Fake Detection Challenge 数据集准备指南

## 步骤

1. 申请访问[Deep Fake Detection Challenge Dataset](https://www.kaggle.com/competitions/deepfake-detection-challenge)的权限，并下载这个数据集。

2. 采用这个[codebase](https://github.com/biubug6/Pytorch_Retinaface)的[脚本](https://github.com/biubug6/Pytorch_Retinaface/blob/master/convert_to_onnx.py)生成retinaface的onnx。

3. 运行下面的命令

```shell
python3 convert_dfdc.py --retinaface <path/to/retinaface/onnx> --dfdc_root <path/to/uncompressed/dfdc/dataset/root> --output <output/directory>
```

## 处理完成的数据结构

```shell
data/
└── dataset.pkl
```

