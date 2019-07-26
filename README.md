[![](https://i.imgur.com/1U9te27.png)](#)

# Cassiopeia Project

This is a Machine Learning Project
Latest Version: Asterion

## Installation

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
participant rNN_WENG
participant ECU
cNN_DT->rNN_WENG: Trend Prediction
cNN_EC->rNN_WENG: Cycle Prediction
rNN_WENG->ECU: Control Signal
Note left of cNN_DT: Driving Trend\nNeural Network
Note left of cNN_EC: Engine Cycle\nNeural Network
Note left of rNN_WENG: Controller\nNeural Network
Note left of ECU: Engine\nControl Unit
```

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
