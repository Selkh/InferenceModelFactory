# ST-GCN preporcessed Dataset Preparation Guide
## Step
st-gcn use author preprocessed kinectic and ntu datasets, which can be downloaded directly.

1. download from https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb
2. `unzip <path to st-gcn-processed-data.zip>`

## Processed Dataset Structure
```
data
|-- Kinetics
|   `-- kinetics-skeleton
|       |-- train_data.npy
|       |-- train_label.pkl
|       |-- val_data.npy
|       `-- val_label.pkl
`-- NTU-RGB-D
    |-- xsub
    |   |-- train_data.npy
    |   |-- train_label.pkl
    |   |-- val_data.npy
    |   `-- val_label.pkl
    `-- xview
        |-- train_data.npy
        |-- train_label.pkl
        |-- val_data.npy
        `-- val_label.pkl
```