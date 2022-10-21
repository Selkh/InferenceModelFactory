# ICDAR 2015 数据集准备指南

## 步骤

1. 下载[ICDAR 2015 测试数据集](https://rrc.cvc.uab.es/?ch=4&com=downloads)(下载需要注册)。注册完登录后，下载“Task 4.1: Text Localization (2015 edition)”中的“Test Set Images”和“Test Set Ground Truth”，其中，Test Set Images下载的内容保存到ch4_test_images文件夹内，Test Set Ground Truth放在ch4_test_localization_transcription_gt文件夹内。

2. 解压下载的压缩文件：

``` shell
cd path/to/ch4_test_images
unzip ch4_test_images.zip
cd path/to/ch4_test_localization_transcription_gt
unzip ch4_test_localization_transcription_gt.zip
```

3. 下载[label 文件](https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt)，将其放到ch4_test_images同目录下。
4. 下载https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json，将其放到ch4_test_images同目录下，并执行python3 modify_directory.py。

## 处理完成的数据结构

```shell
data/
├── ch4_test_images
│        ├── img_1.jpg
│        ├── img_2.jpg
|        └── ……
├── ch4_test_localization_transcription_gt
│        ├── gt_img_1.txt
│        ├── gt_img_2.txt
|        └── ……
├── instances_test.json
└── test_icdar2015_label.txt
```

