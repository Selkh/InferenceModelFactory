# LFW 数据集准备指南

## 步骤

1. 下载[facenet](https://github.com/davidsandberg/facenet)代码

2. 下载[LFW](http://vis-www.cs.umass.edu/lfw/lfw.tgz), [pairs.txt](http://vis-www.cs.umass.edu/lfw/pairs.txt)，并且解压缩

3. 执行下面的命令

```shell
cd <path/to/facenet>
pip3 install -r requirements.txt
for N in {1..4}
do
PYTHONPATH=src python3 src/align/align_dataset_mtcnn.py <path/to/uncompressed/lfw/directory> <path/to/output/directory> --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25
done
mkdir data
cp -r <path/to/output/directory> data/lfw
cp <path/to/pairs.txt> data
```

## 处理完成的数据结构

```shell
data/
├── lfw
└── pairs.txt
```

