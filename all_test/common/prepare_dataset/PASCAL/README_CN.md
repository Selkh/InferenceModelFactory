# PASCAL 数据集准备
## 步骤

1. 下载 [PASCAL](https://drive.google.com/open?id=1p7dDQgYh2RBPUZSlOQVU4PgaSKlq64ik)
2. 解压pascal_images.zip文件 `unzip -d <data/PASCAL> <pascal_images.zip>`, `mv data/PASCAL/pascal_images data/PASCAL/images`，目录结构如下：
    ```shell
    data
        └── PASCAL
            └── images
                ├── 2007_000272.jpg
                ├── 2007_000664.jpg
                ├── ...
                ├── 2011_003272.jpg
                └── 2011_003273.jpg
    ```