# kitti Dataset Preparation Guide
## Step


Download kitti-dataset form [kaggle](https://www.kaggle.com/datasets/klemenko/kitti-dataset/download?datasetVersionNumber=1)  or [kitti official website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), then uncompress it, the directory structure lookslike:

```shell
    data/
        └── kitti/
            ├──ImageSets/
            │   ├── test.txt
            │   ├── train.txt
            │   └── val.txt
            ├── training/
            │   ├── image_2/
            │   ├── calib/
            │   ├── label_2/
            │   └── velodyne/
            ├── testing/
            │   ├── image_2/
            │   ├── calib/
            │   └── velodyne/
            └── classes_names.txt
```