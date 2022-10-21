# voxceleb1 voxceleb2 dataset preparation guide

## step

1. Download the Dev files and Test files of [voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) from the official website. Note that you need to apply for a download account yourself. If the official website download link is invalid, you can use <https://mm.kaist.ac.kr/datasets/voxceleb/>.

2. Download all files and concatenate into zip file

   ```
   cat vox1_dev* > vox1_dev_wav.zip
   ```

3. Download the Dev files and Test files of [voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) from the official website. Note that you need to apply for a download account yourself. If the official website download link is invalid, you can use <https://mm.kaist.ac.kr/datasets/voxceleb/>.

4. Download all files and concatenate into zip file

   ```
   cat vox2_dev_aac* > vox2_aac.zip
   ```

5. After the download is complete, there will be four zip files vox2_test_mp4.zip vox2_aac.zip vox1_test_wav.zip vox1_dev_wav.zip.

6. Execute the following command

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

## processed data structure

   ```shell
   vox1_2
   ├── convert.sh
   └── wav
      ├── id00012
         ....
      └── id11251
   ```
