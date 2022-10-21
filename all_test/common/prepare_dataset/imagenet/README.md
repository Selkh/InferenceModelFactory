# Imagenet Dataset Preparation Guide
## Step
1. download ILSVRC2012_img_val.tar from https://image-net.org/challenges/LSVRC/2012/ (you need register)
2. extract
    ```bash
    mkdir val
    tar -xvf ILSVRC2012_img_val.tar -C val/
    ```
3. download labels
    ```bash
    wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
    wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt
    ```
4. put images into category folders (if a flatten dir structure is needed, skip)
    ```bash
    python3 preprocess_imagenet_validation_data.py val/ imagenet_2012_validation_synset_labels.txt imagenet_lsvrc_2015_synsets.txt
    cp imagenet_2012_validation_synset_labels.txt val/synset_labels.txt
    ```
5. generate val_map.txt
    ```bash
    python3 convert_imagenet.py val/ imagenet_2012_validation_synset_labels.txt imagenet_lsvrc_2015_synsets.txt val/val_map.txt
    ```
6. rename
    ```bash
    mv val data
    ```
## Processed Dataset Structure
```
data
   ├── n01440764
   │   ├── ILSVRC2012_val_00000293.JPEG
   │   ├── ILSVRC2012_val_00002138.JPEG
   |   └── ……
   ……
   └── val_map.txt
```

val_map.txt contains image path and label relationship likes:

```
./n01751748/ILSVRC2012_val_00000001.JPEG 65
./n09193705/ILSVRC2012_val_00000002.JPEG 970
./n02105855/ILSVRC2012_val_00000003.JPEG 230
./n04263257/ILSVRC2012_val_00000004.JPEG 809
……
```