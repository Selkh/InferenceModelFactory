# mnist 数据集准备指南

## 步骤

1. 从官网下载[mnist](http://yann.lecun.com/exdb/mnist/)数据集。

   ```shell
   wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
   wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
   wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
   wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
   ```

## 处理完的数据结构

   ```shell
      data/
      ├── t10k-images-idx3-ubyte.gz
      ├── t10k-labels-idx1-ubyte.gz
      ├── train-images-idx3-ubyte.gz
      └── train-labels-idx1-ubyte.gz
   ```
