# Widerface Dataset Preparation Guide
## Step

1. Download the following files and put them under one directory.

| file | url |
|----|----|
| WIDER_val.zip | https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view?usp=sharing |
| wider_easy_val.mat | https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/wider_easy_val.mat |
| wider_face_val.mat | https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/wider_face_val.mat |
| wider_hard_val.mat | https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/wider_hard_val.mat |
| wider_medium_val.mat | https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/wider_medium_val.mat |

2. Run commands below

```shell
pip3 install -r requirements.txt
python3 convert_widerface.py --input_path=<path/to/the/directory/containing/the/three/files> --output_path=<path/to/data>
```

## Processed Dataset Structure

```shell
data/
├── annotations
└── WIDER_val
```

