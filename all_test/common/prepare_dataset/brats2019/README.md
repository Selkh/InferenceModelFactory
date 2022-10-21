# brats2019 preporcessed Dataset Preparation Guide
## Step

1. Download [archive.zip](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019). Put it under a directory.
2. `unzip -d <data/> <archive.zip>`, the directory structure looks like:

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

3. `gzip -r data/MICCAI_BraTS_2019_Data_Training/HGG/*`, `gzip -r data/MICCAI_BraTS_2019_Data_Training/LGG/*`, compress .nii files into .nii.gz, the directory structure looks like:

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

4. Download the [model](https://zenodo.org/record/3904106/files/fold_1.zip) and `unzip -d <data/> <fold_1.zip>` and for later data preprocessing, the directory structure looks like:

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