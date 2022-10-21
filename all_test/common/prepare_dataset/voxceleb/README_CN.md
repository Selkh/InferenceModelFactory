# voxceleb1 voxceleb2 数据集准备指南

## 步骤

1. 从官网下载[voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)的Dev文件和Test文件.。注意需要自己申请下载账号。如果官网下载链接失效了，可以使用<https://mm.kaist.ac.kr/datasets/voxceleb/>。

2. 下载所有文件并拼接成zip文件

   ```
   cat vox1_dev* > vox1_dev_wav.zip
   ```

3. 从官网下载[voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)的Dev文件和Test文件.。注意需要自己申请下载账号。如果官网下载链接失效了，可以使用<https://mm.kaist.ac.kr/datasets/voxceleb/>。

4. 载所有文件并拼接成zip文件

   ```
   cat vox2_dev_aac* > vox2_aac.zip
   ```

5. 下载完成后会有vox2_test_mp4.zip vox2_aac.zip vox1_test_wav.zip vox1_dev_wav.zip四个zip文件。

6. 执行以下命令

   ```
   mkdir -p vox1_2/wav
   unzip vox1_dev_wav.zip -d vox1
   unzip vox1_test_wav.zip -d vox1

   unzip vox2_aac.zip -d vox2

   cp -r vox2/dev/aac/id*  vox1_2/wav
   cp -r vox1/wav/id* vox1_2/wav

   cp convert.sh vox1_2
   cd vox1_2
   bash ./convert.sh
   ```

## 处理完的数据结构

   ```shell
   vox1_2
   ├── convert.sh
   └── wav
      ├── id00012
         ....
      └── id11251
   ```
