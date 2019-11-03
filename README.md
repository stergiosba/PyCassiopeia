[![](https://i.imgur.com/1U9te27.png)](#)

# Cassiopeia Project

This is a Machine Learning Project
Latest Version: Asterion v1.0

PROJECT IS MIGRATING TO C++/Python

## Installation

This is for Asterion Version which will stay pure python and will be transfered to a different project in the future.

After you clone the latest version which should be master follow these instuctions:

Since the project uses tensorflow there are two Conda environments available

To install the current release for CPU-only:

```
conda env create -f cassiopeia.yaml
```
To install the current release for GPU:
```
conda env create -f cassiopeia_gpu.yaml
```

###Sequence Diagram
                    
```sequence
participant cNN_DT
participant cNN_EC
participant rNN_ENG
participant ECU
cNN_DT->rNN_WENG: Trend Prediction
cNN_EC->rNN_WENG: Cycle Prediction
rNN_ENG->ECU: Control Signal
Note left of cNN_DT: Driving Trend\nNeural Network
Note left of cNN_EC: Engine Cycle\nNeural Network
Note left of rNN_ENG: Controller\nNeural Network
Note left of ECU: Engine\nControl Unit
```

## Acknowledgments

* Tensorflow/Pandas/Numpy
* Code for my Master Diploma Thesis
