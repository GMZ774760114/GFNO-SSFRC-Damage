# GFNO for Anisotropic Damage Prediction in SSFRC

This repository contains the dataset and training code for a Graph Fourier Neural Operator (GFNO) model for predicting the anisotropic damage tensor of short steel fiber reinforced concrete (SSFRC).

## Structure

data/  
└── dataset/ (4320 samples)

code/  
└── train_gfno.py

splits/  
├── train.txt  
├── val.txt  
└── test.txt  

configs/  
└── gfno.yaml

## Dataset

- 4320 samples (.npz)
- Each sample includes:
  - segments  
  - label_D6  
  - global_features  

See `data/README_data.md` for details.

## Training

```bash
python code/train_gfno.py