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
<img width="8192" height="5469" alt="Picture1" src="https://github.com/user-attachments/assets/8a67997c-2480-43d6-bb8a-121319eb9d67" />
<img width="8192" height="8192" alt="Picture2" src="https://github.com/user-attachments/assets/dc4b8852-44f7-4a99-9d27-3283ec314d6c" />

