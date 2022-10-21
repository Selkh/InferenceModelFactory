# AFW preporcessed Dataset Preparation Guide
## Step

1. Download the images of [AFW](https://drive.google.com/open?id=1Kl2Cjy8IwrkYDwMbe_9DVuAwTHJ8fjev)
2. `unzip -d <data/AFW> <afw_images.zip>`, `mv data/AFW/afw_images data/AFW/images`, the directory structure looks like:
    ```shell
    data
        └── AFW
            └── images
                ├── 1004109301.jpg
                ├── 1051618982.jpg
                ├── ...
                ├── README
                └── anno.mat
    ```