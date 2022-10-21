# PASCAL preporcessed Dataset Preparation Guide
## Step

1. Download the images of [PASCAL](https://drive.google.com/open?id=1p7dDQgYh2RBPUZSlOQVU4PgaSKlq64ik)
2. `unzip -d <data/PASCAL> <pascal_images.zip>`, `mv data/PASCAL/pascal_images data/PASCAL/images`, the directory structure looks like:
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