# DelicacyNet

## Introduction

DelicacyNet is an image processing model for food content prediction. In this paper, we present its structure, which consists of four layers. We used semantic understanding and then converted to vector modules and final data correspondences, so the degree of our semantic approximation will affect the result of the corresponding prediction. You can replace the encoder module and the decoder module, as the other modules are relatively independent and still achieve good performance.

![model](https://github.com/nerakoo/DelicacyNet/blob/main/datasets/model.png)



## Installation

Our code is tested with Python>=3.6/3.7/3.8, PyTorch>=1.6.0/1.7.0/1.9.0, CUDA==10.2 on Ubuntu-18.04 with NVIDIA GeForce RTX 2080Ti. Similar or higher version should work well.



We highly recommend using [Anaconda](https://www.anaconda.com/) to manage the python environment:

```
conda create -n transmvsnet python=3.6
conda activate transmvsnet
pip install -r requirements.txt
```



## Data preparation

You can email the Food2K authors to obtain the database license and download address. In addition, the Food2K label can be found in their [original paper](http://123.57.42.89/Large-Scale_Food_Recognition_via_Deep_Progressive_Self-Transformer_Network/Supplementary%20tables.pdf).

A download of the USDA database can be found in their [original paper](https://data.nal.usda.gov/dataset/usda-national-nutrient-database-standard-reference-legacy-release).



## Training the model

If we want to train our own model, we can use the following code:

```python
python main.py
```



## Viewing the process

The project contains the details of the training process and is stored in a specific file. Here is a list of traditional viewing methods:

```bash
cd ./DelicacyNet/output/log.txt
```



## Acknowledgments

Special thanks go to the providers of databases like Food2K, USDA, and Recipe1M+ for their great contributions, as well as to ASPP, the authors of the transformer paper, from whom we borrowed parts of the code.
