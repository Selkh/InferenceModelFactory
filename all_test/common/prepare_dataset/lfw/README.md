# LFW Dataset Preparation Guide

## Step

1. download code base [here](https://github.com/davidsandberg/facenet)

2. download [lfw](http://vis-www.cs.umass.edu/lfw/lfw.tgz), [pair list](http://vis-www.cs.umass.edu/lfw/pairs.txt) and uncompress it.

3. Run commands below

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

## Processed Dataset Structure

```shell
data/
├── lfw
└── pairs.txt
```

