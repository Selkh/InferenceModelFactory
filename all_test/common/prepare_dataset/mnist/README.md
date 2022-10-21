# mnist dataset preparation guide

## step

1. Download the Dev files and Test files of [mnist](http://yann.lecun.com/exdb/mnist/)

   ```shell
   wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
   wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
   wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
   wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
   ```

## processed data structure

   ```shell
      data/
      ├── t10k-images-idx3-ubyte.gz
      ├── t10k-labels-idx1-ubyte.gz
      ├── train-images-idx3-ubyte.gz
      └── train-labels-idx1-ubyte.gz
   ```
