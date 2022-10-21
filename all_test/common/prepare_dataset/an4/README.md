# AN4 Dataset Preparation Guide
## Step
1. Download [an4](http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz). Put it under a directory.

2. Run commands below

```shell
pip3 install -r requirements.txt
python3 sph2wav.py --dir_path=<path/to/the/directory/you/containing/an4_sphere.tar.gz>
python3 build_mainfest.py --dataset_path=<path/to/the/directory/you/containing/an4test_clstk> --dir_path=<path/to/the/directory/you/output/test_manifest.json>
```

As far as we know, the link of `an4` is currently unaccessible. 

## Processed Data Structure

```shell
data/
└── an4
     ├── etc
     |     ├──an4_test.transcription
     |     └── ...
     ├── wav
     |    ├── an4_clstk
     |    └── an4test_clstk
     └──  test_manifest.json
```

