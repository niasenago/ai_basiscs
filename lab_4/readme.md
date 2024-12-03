https://www.youtube.com/watch?v=6_2hzRopPbQ

Prerequisuites:
You must download:
https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification/data

To check your cuda driver and tensorflow lib compatability go here:
https://www.tensorflow.org/install/source#gpu
---
We define architecture of our network in `hyperparams.json` file (in `layers` list). `main.py` script fetches this file and builds model acording to defined architecture.

`hyperparams.json` file example:
```json
{
    "layers": [
      {
        "units": 32,
        "activation": "relu"
      },
      {
        "units": 64,
        "activation": "relu"
      },
      {
        "units": 1,
        "activation": "sigmoid"
      }
    ]
}
  
```
---


| Model   | Dropout Layer(s)                   | Batch Normalization            | Activation Function | Optimizer |
|---------|------------------------------------|---------------------- ---------|---------------------|-----------|
| Model 1 | 1 Dropout layer                    | No batch normalization         | ReLU                | SGD       |
| Model 2 | No dropout layer                   | No batch normalization         | ReLU                | SGD       |
| Model 3 | 1 Dropout whith anther probability | No batch normalization         | ReLU                | SGD       |
| Model 4 | 1 Dropout layer                    | Batch normalization after each Conv2D | ReLU         | SGD       |
| Model 5 | 1 Dropout layer                    | No batch normalization         | Tanh                | SGD       |
| Model 6 | 1 Dropout layer                    | No batch normalization         | ReLU                | Adam      |
