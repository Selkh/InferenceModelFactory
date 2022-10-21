# Kinetics400 数据集准备指南

## 步骤

1. 下载Kinetics评测视频压缩文件[kinetics_400_val_320.tar](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB).

2. 下载Kinetics400标注文件[kinetics_val_list.txt](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt).

3. 解压缩 kinetics_400_val_320.tar 并且把文件夹 kinetics_400_val_320 重命名为 val_320

3. 组织上述文件为如下结构

## 处理完成的数据结构

```shell
data/
├── label
│     └── kinetics_val_list.txt
└── val_320
```

