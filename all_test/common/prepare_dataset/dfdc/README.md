# Deep Fake Detection Challenge Dataset Preparation Guide

## Step

1. Apply for admission to [Deep Fake Detection Challenge Dataset](https://www.kaggle.com/competitions/deepfake-detection-challenge) and download it.

2. Generate retinaface onnx from this [codebase](https://github.com/biubug6/Pytorch_Retinaface) with its [script](https://github.com/biubug6/Pytorch_Retinaface/blob/master/convert_to_onnx.py).

3. Run commands below

```shell
python3 convert_dfdc.py --retinaface <path/to/retinaface/onnx> --dfdc_root <path/to/uncompressed/dfdc/dataset/root> --output <output/directory>
```

## Processed Dataset Structure

```shell
data/
└── dataset.pkl
```

