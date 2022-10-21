# Widerface 数据集准备指南
## 步骤

1. 下载下面的文件并放到一个文件夹下面。

| file | url |
|----|----|
| WIDER_val.zip | https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view?usp=sharing | 
| wider_face_split.zip | http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip |
| retinaface_gt_v1.1.zip | https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA |
| wider_easy_val.mat | https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/wider_easy_val.mat |
| wider_face_val.mat | https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/wider_face_val.mat |
| wider_hard_val.mat | https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/wider_hard_val.mat |
| wider_medium_val.mat | https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/wider_medium_val.mat |

2. 运行下面的命令

```shell
pip3 install -r requirements.txt
python3 convert_widerface.py --input_path=<你刚创建的文件夹> --output_path=<目标路径>
```

## 处理完的数据结构

```shell
data/
├── annotations
└── WIDER_val
```

