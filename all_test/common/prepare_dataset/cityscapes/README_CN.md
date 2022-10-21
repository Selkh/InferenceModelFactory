# CITYSCAPES 数据集准备

## 步骤

1. 首先您需要在[cityscapes官网](https://www.cityscapes-dataset.com/register/)注册一个账号，并激活、登录此账号
2. 在该[网页](https://www.cityscapes-dataset.com/downloads/)点击下载 [dataset](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and [annotations](https://www.cityscapes-dataset.com/file-handling/?packageID=1) , 标注文件名是[gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1), 数据集名称是[leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
3. 解压在第2步中下载得到的2个文件，最终使数据集目录结构如下：

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
4. 通常情况下，`**labelTrainIds.png` 被用来训练 cityscapes。 基于 [cityscapesscripts](https://github.com/mcordts/cityscapesScripts), open-mmlab提供了一个 [脚本](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py), 去生成 `**labelTrainIds.png`。你可以参考[此网页](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/dataset_prepare.md)去生成合适的cityscapes的labels。注意，在您能生成上述labels之前，您可能需要先安装mmcv和mmsegmentation，您可以参考这个[安装向导](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/get_started.md)。
