# CITYSCAPES Dataset Preparation Guide

## Step

1. First of all, you need to register an account in this [webpage](https://www.cityscapes-dataset.com/register/) , then you should activate and login your account
2. click to download [dataset](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and [annotations](https://www.cityscapes-dataset.com/file-handling/?packageID=1) at this [page](https://www.cityscapes-dataset.com/downloads/), the annotations is [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1), the dataset is [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
3. unzip those two files which you downloaded in step 2, the directory structure looks like:

```shell
cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   └── val
└── leftImg8bit
    ├── test
    ├── train
    └── val
```

4. By convention, `**labelTrainIds.png` are used for cityscapes training.  Open-mmlab provided a [scripts](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py) based on [cityscapesscripts](https://github.com/mcordts/cityscapesScripts) to generate `**labelTrainIds.png`., you can refer to this [tutorial](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) to generate cityscapes labels. Attention, you may need to install mmcv and mmsegmentation before you can generate above labels, you could refer to this[ install guide.](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation)
