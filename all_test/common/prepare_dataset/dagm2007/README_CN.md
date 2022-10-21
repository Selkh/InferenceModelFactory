# dagm2007 数据集准备指南
## 步骤

1. 在该网页 https://hci.iwr.uni-heidelberg.de/node/3616 中注册账户
2. 注册完后获得下载链接，下载Class1.zip类别1的数据
3. 解压到**dagm2007/private/**文件夹，`unzip -d <dagm2007/private/> <Class1.zip>`, 处理完成后的数据结构如下:

    ```shell
    dagm2007
        └── private
                └── Class1
                    |── Test
                    |   └── Label
                    |   |   |── 0002_label.PNG
                    |   |   |── ...
                    |   |   |── 0568_label.PNG
                    |   |   └── Labels.txt
                    |   |── 0001.PNG
                    |   |── ...
                    |   └── 0575.PNG
                    |── ...
                    └── test_list.csv
    ```

4. 运行命令生成test_list.csv：python3 preprocess_dagm2007.py --data_dir=dagm2007/private/
