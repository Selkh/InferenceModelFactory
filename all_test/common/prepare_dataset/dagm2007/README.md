# dagm2007 preporcessed Dataset Preparation Guide
## Step

1. Create an account on 'https://hci.iwr.uni-heidelberg.de/node/3616'
2. Download the Class1.zip file
3. `unzip -d <dagm2007/private/> <Class1.zip>`, the directory structure looks like:

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

4. python3 preprocess_dagm2007.py --data_dir=dagm2007/private/
