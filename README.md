Ocean pCO2 Prediction with Deep Learning
==============================

Author: Shaun (Siyeon) Kim, [sk4973@columbia.edu](mailto:sk4973@columbia.edu)

### Global Ocean pCO2 modeling for Gentine Lab

This repository contains models that predict pCO2 level of the ocean by incorporating spatial and temporal information with the help of Deep Learning (Image Segmentations + ConvLSTM algorithms). 

It also contains traditional machine learning models such as neural network, random forest, and XGboost.  

### CNN-UNET Pretrained Model

**PERFORMANCE** 

![Alt text](https://github.com/sk981102/ocean_co2/blob/main/assets/cnn-unet.gif)

**RMSE over TIME** 

![Alt text](https://github.com/sk981102/ocean_co2/blob/main/assets/unet-overtime.png)


### ConvLSTM Pretrained Model 

**PERFORMANCE** 

![Alt text](https://github.com/sk981102/ocean_co2/blob/main/assets/cnn-lstm.gif)

**RMSE over TIME** 

![Alt text](https://github.com/sk981102/ocean_co2/blob/main/assets/nfp-overtime.png)


#### Performance compared to Traditional ML

| Model  | RMSE (uatm) |
| ------------- | ------------- |
| Random Forest  | 40.387 |
| FFN  | 39.494 |
| XgBoost  | 37.709  |
| **CNN-UNET**  | **8.499** |
| **ConvLSTM** | **3.737**  |


### Potenial Use Cases of Pretrained Models

1. Used to predict dpCO2 in addition to pCO2 via transfer learning
2. Used to predict pCO2 in real world SOCAT sampling via transfer learning

Getting Started
------------

### Requirements
This model was trained on the following libraries:

```` 
cuda11.0/toolkit cuda11.0/blas cudnn8.0-cuda11.
tensorflow==2.4.0
````

### Downloading Data and Libraries
To download the data from [figshare](https://figshare.com/articles/dataset/CESM_ocean_pCO2_testbed/8798999?file=16129505):
```` 
mkdir data #create data directory
./download_data.sh [file_id]
````
**file_id : 8 code digit at the end of the data url from figshare**


To download the python libraries:
```` 
pip install -r requirements.txt
````

### Colab Notebook for Run-through




Project Organization
------------
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks. Consists of EDA and Base Model implementations.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── utils.py       <- various util functions for data preprocessing and plotting
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    └── 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.
