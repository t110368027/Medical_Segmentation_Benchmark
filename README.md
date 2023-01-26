
---

<div align="center">    
 
# Medical Segmentation Benchmark



[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-blue?style=for-the-badge&logo=python)](https://docs.python.org/3.7/)
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.12.1-red?style=for-the-badge&logo=pytorch)](https://github.com/pytorch/pytorch)
[![Pytorch-Lightning - Version](https://img.shields.io/badge/pytorch_Lightning-1.9.0+-%3CCOLOR%3E.svg?style=for-the-badge&logo=pytorch-lightning&logoColor=green)](https://github.com/Lightning-AI/lightning)

[![MONAI - Version](https://img.shields.io/badge/Monai-1.1.0+-blue?style=for-the-badge)](https://github.com/Project-MONAI/MONAI)
[![Albumentations - Version](https://img.shields.io/badge/albumentations-1.3.0+-red?style=for-the-badge)](https://github.com/albumentations-team/albumentations/)

</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/t110368027/Medical_Segmentation_Benchmark

# install project   
cd Medical_Segmentation_Benchmark
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# run module   
python train.py
```

## Prepare dataset
Before run this project you need to download the dataset :
```bash
./Medical_Segmentation_Benchmark/
├── data/
│   ├─angiography
│   │  ├─Hemotool
│   │  │  └─angio1.png_mask.png  
│   │  ├─Original
│   │  │  └─1.png
│   │  └─Photoshop
│   │     └─angio1ok.png
│   ├─CHASE_DB1
│   │  ├─Images
│   │  │  └─Image_01L.jpg
│   │  └─Masks
│   │     └─Image_01L_1stHO.png
│   ├─DB_Angiograms_134
│   │  ├─Database_134_Angiograms
│   │  │  ├─1.pgm
│   │  │  ├─1_gt.pgm
│   ├─STARE
│   │  ├─imgs
│   │  │  └─im0001.png
│   │  └─label-ah
│         └─im0001.ah.png
├── datasets/
│   ├─CHASEDB1
│   │  └─set.npz
│   ├─CHUAC
│   │  └─set.npz
│   ├─DCA1
│   │  └─set.npz
│   └─STARE
│      └─set.npz
├── README.md
└── train.py
```
Convert `.jpg`, `.png` or `.pgm` to numpy array and save it to `.npz` file
 ```bash
# run module   
python preprocess.py
```

## Citation   
```
@misc{Medical_Segmentation_Benchmark,
  author = {Jia-Ming Hou},
  title = {{Medical_Segmentation_Benchmark}},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/t110368027/Medical_Segmentation_Benchmark}},
  version = {0.0.1}, 
}
```   
## License
 
Project is distributed under [Apache 2.0](https://github.com/t110368027/Medical_Segmentation_Benchmark/blob/main/LICENSE)