# brats2019 数据集准备指南
## 步骤

1. 下载 [archive.zip](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019)。 把它放到一个文件夹下面。
2. 解压到**data/**文件夹， `unzip -d <data/> <archive.zip>`， 目录结构如下:

    ```shell
    data/
        └── MICCAI_BraTS_2019_Data_Training
            |-- HGG
            |   |-- BraTS19_2013_10_1
            |   |   |-- BraTS19_2013_10_1_flair.nii
            |   |   |-- BraTS19_2013_10_1_seg.nii
            |   |   |-- BraTS19_2013_10_1_t1.nii
            |   |   |-- BraTS19_2013_10_1_t1ce.nii
            |   |   └── BraTS19_2013_10_1_t2.nii
            |   |-- BraTS19_2013_11_1
            |   └── ...
            |-- LGG
            |   |-- BraTS19_2013_0_1
            |   |   |-- BraTS19_2013_0_1_flair.nii
            |   |   |-- BraTS19_2013_0_1_seg.nii
            |   |   |-- BraTS19_2013_0_1_t1.nii
            |   |   |-- BraTS19_2013_0_1_t1ce.nii
            |   |   └── BraTS19_2013_0_1_t2.nii
            |   |-- BraTS19_2013_15_1
            |   └── ...
            |── name_mapping.csv
            └── survival_data.csv
    ```

3. 将**data/MICCAI_BraTS_2019_Data_Training**文件夹中的所有.nii文件压缩成.nii.gz文件格式, `gzip -r data/MICCAI_BraTS_2019_Data_Training/HGG/*`, `gzip -r data/MICCAI_BraTS_2019_Data_Training/LGG/*`， 目录结构如下:

    ```shell
    data/
        └── MICCAI_BraTS_2019_Data_Training
            |-- HGG
            |   |-- BraTS19_2013_10_1
            |   |   |-- BraTS19_2013_10_1_flair.nii.gz
            |   |   |-- BraTS19_2013_10_1_seg.nii.gz
            |   |   |-- BraTS19_2013_10_1_t1.nii.gz
            |   |   |-- BraTS19_2013_10_1_t1ce.nii.gz
            |   |   └── BraTS19_2013_10_1_t2.nii.gz
            |   |-- BraTS19_2013_11_1
            |   └── ...
            |-- LGG
            |   |-- BraTS19_2013_0_1
            |   |   |-- BraTS19_2013_0_1_flair.nii.gz
            |   |   |-- BraTS19_2013_0_1_seg.nii.gz
            |   |   |-- BraTS19_2013_0_1_t1.nii.gz
            |   |   |-- BraTS19_2013_0_1_t1ce.nii.gz
            |   |   └── BraTS19_2013_0_1_t2.nii.gz
            |   |-- BraTS19_2013_15_1
            |   └── ...
            |── name_mapping.csv
            └── survival_data.csv
    ```

4. 提供数据预处理相关模型文件， 下载 [模型](https://zenodo.org/record/3904106/files/fold_1.zip) 并解压到**data/**文件夹， `unzip -d <data/> <fold_1.zip>`， 目录结构如下:

    ```shell
    data/
        |-- MICCAI_BraTS_2019_Data_Training
        |   |-- HGG
        |   |   |-- BraTS19_2013_10_1
        |   |   |   |-- BraTS19_2013_10_1_flair.nii.gz
        |   |   |   |-- BraTS19_2013_10_1_seg.nii.gz
        |   |   |   |-- BraTS19_2013_10_1_t1.nii.gz
        |   |   |   |-- BraTS19_2013_10_1_t1ce.nii.gz
        |   |   |   └── BraTS19_2013_10_1_t2.nii.gz
        |   |   |-- BraTS19_2013_11_1
        |   |   └── ...
        |   |-- LGG
        |   |   |-- BraTS19_2013_0_1
        |   |   |   |-- BraTS19_2013_0_1_flair.nii.gz
        |   |   |   |-- BraTS19_2013_0_1_seg.nii.gz
        |   |   |   |-- BraTS19_2013_0_1_t1.nii.gz
        |   |   |   |-- BraTS19_2013_0_1_t1ce.nii.gz
        |   |   |   └── BraTS19_2013_0_1_t2.nii.gz
        |   |   |-- BraTS19_2013_15_1
        |   |   └── ...
        |   |── name_mapping.csv
        |   └── survival_data.csv
        |-- nnUNet
        |   └── 3d_fullres
        |       └── Task043_BraTS2019
        |           └── nnUNetTrainerV2__nnUNetPlansv2.mlperf.1
        |               |-- fold_1
        |               |   |-- debug.json
        |               |   |-- model_best.model
        |               |   |-- model_best.model.pkl
        |               |   |-- model_final_checkpoint.model
        |               |   |-- model_final_checkpoint.model.pkl
        |               |   |-- postprocessing.json
        |               |   |-- progress.png
        |               |   |-- training_log_2020_5_25_19_07_42.txt
        |               |   |-- training_log_2020_6_15_14_50_42.txt
        |               |   └── training_log_2020_6_8_08_12_03.txt
        |               └── plans.pkl
        └── joblog.log
    ```