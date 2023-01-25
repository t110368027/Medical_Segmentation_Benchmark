
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
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
