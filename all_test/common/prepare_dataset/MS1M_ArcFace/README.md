# Face MS1M validation data Preparation Guide

## Step

1. Download [faces_ms1m_112x112.zip](https://s3.amazonaws.com/onnx-model-zoo/arcface/dataset/faces_ms1m_112x112.zip). Put it under a directory.

2. Run commands below

```shell
# suggest python3.7
pip3 install -r requirements.txt
python3 convert_ms1m_face.py --input_data_dir=<path/to/the/directory/containing/the/file> --output_data_dir=<path/to/the/converted/data>
```

## Processed Dataset Structure

```
converted_ms1m_face
   ├── agedb_30.bin
   ├── cfp_ff.bin
   ├── cfp_fp.bin
   └── lfw.bin
```
