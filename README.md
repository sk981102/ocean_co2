Ocean pCO2 Prediction with Deep Learning
==============================

Author: Shaun (Siyeon) Kim, [sk4973@columbia.edu](mailto:sk4973@columbia.edu)

### Global Ocean pCO2 modeling for Gentine Lab

This repository contains models that predict pCO2 level of the ocean by incorporating spatial and temporal information with the help of Deep Learning. It also contains traditional machine learning models such as neural network, random forest, and XGboost.  

#### CNN Image Segmentation Output
**Predicted vs. Expected**

![Alt text](https://github.com/sk981102/ocean_co2/blob/main/assets/val_comparison.png)

**Residual**

![Alt text](https://github.com/sk981102/ocean_co2/blob/main/assets/val_residual.png)

**RMSE over time**

![Alt text](https://github.com/sk981102/ocean_co2/blob/main/assets/val_overtime.png)


Getting Started
------------

### Requirements
This model was trained on the following libraries:

```` 
cuda11.0/toolkit cuda11.0/blas cudnn8.0-cuda11.
tensorflow==2.4.0
````

### Data and Model Usage
To download the data from figshare & download python libraries:
```` 
python3 setup.py
````

To download the python libraries:
```` 
pip install -r requirements.txt
````

Usage:
```python

import sys
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, '../src')
from utils import preprocess

model_dir="{{model_dir}}"
data = preprocess({{data}}) # write the wrapper function for preprocessing data

best_model = tf.keras.models.load_model(model_dir, custom_objects={'custom_rmse':custom_rmse})

predicted_images=best_model.predict(data,verbose=1)

```

### Colab Notebook for Run-through

[Run the model on Google Colab (interactive notebook)](https://colab.research.google.com/drive/1GX6e0XXeGSb7F4sZDQwSW4cA7dwgMc7I)



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
