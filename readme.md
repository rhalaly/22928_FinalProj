# PoseNet

PoseNet – An accurate neural network for head pose estimation.

## Background

PoseNet is an accurate neural network based on the principals of [HopeNet](https://github.com/natanielruiz/deep-head-pose). There are two types of PoseNet for different scenarios and training protocols.

1. PoseNet - A model that has been trained on datasets with a labeled pose, that tries to compete with other models on the common protocol for head pose estimation problem.
2. PoseNet_pnp - A model that has been trained on the results of the PnP algorithm that has run on labeled landmarks, that tries to compete with the landmarks + PnP based algorithms.

Each type of PoseNet has 4 trained models that can be found [here](https://drive.google.com/drive/folders/1_nRb12xVTLbrSYIVycKgguqwJL1gsI8X?usp=sharing). Please download the models and put them in the `models` directory.

## Code

- `tester.py` - Runs the models on the final test set.
- `evaluate.py` - Estimates the model on the AFLW2000 dataset.
- `validate2.py` - Estimates the model on the second validation set.
- `posenet.py` - The PoseNet model structure and the predictor.
- `datasets.py` - An interface for the 300W-LP (labeled pose and PnP) and AFLW2000 datasets.
- `utils.py` - Utils and helper functions.
- `validate.py` - [Obsolete] Estimates the model on the first validation set.
    - The code is obsolete, the constants are hard-coded.
- `output` - The output directory that contains the final test set.
  - [`posenet`](https://github.com/rhalaly/22928_FinalProj/tree/master/output/posenet) - Contains all the images from the final test with drawn axis and the [CSV](https://github.com/rhalaly/22928_FinalProj/blob/master/output/posenet/output.csv) file that have been estimated with the **PoseNet** model.
  -  [`posenet_pnp`](https://github.com/rhalaly/22928_FinalProj/tree/master/output/posenet_pnp) - Contains all the images from the final test with drawn axis and the [CSV](https://github.com/rhalaly/22928_FinalProj/blob/master/output/posenet_pnp/output.csv) file that have been estimated with the **PoseNet_pnp** model.

### Command lines

- To run **PoseNet** on the **test set**:
  ```
  python tester.py --posenet_type 0 --output_dir output\posenet\ --models_dir models\ --dataset_dir <TEST_SET_DIR_PATH>
  ```
- To run **PoseNet_pnp** on the **test set**:
  ```
  python tester.py --posenet_type 1 --output_dir output\posenet_pnp\ --models_dir models\ --dataset_dir <TEST_SET_DIR_PATH>
  ```
- To run **PoseNet** on the **Second validation set**:
  ```
  python validate2.py --posenet_type 0 --output_dir output\validation\ --models_dir models\ --dataset_dir <TEST_SET_DIR_PATH>
  ```
- To run **PoseNet** on the **Second validation set**:
  ```
  python validate2.py --posenet_type 1 --output_dir output\validation\ --models_dir models\ --dataset_dir <TEST_SET_DIR_PATH>
  ```

  Note that `tester.py` and `validation2.py` don't create the output directory automatically. To create the images please give a valid path as `--output_dir`.

## Results

The results of **PoseNet** can be found [here](https://github.com/rhalaly/22928_FinalProj/tree/master/output/posenet) and the results of **PoseNet_pnp** can be found [here](https://github.com/rhalaly/22928_FinalProj/tree/master/output/posenet_pnp).

### PoseNet

PoseNet has trained on the dataset **300W-LP** and tested on **AFLW2000**, which is a common protocol for the head pose estimation problem.

There are 4 different PoseNet models, the difference between them is described in the paper.

The MAE of PoseNet on the AFLW2000 dataset:

| Model | MAE Yaw | MAE Pitch | MAE Roll | MAE Total |
|---|---|---|---|---|
| HopeNet (α = 1) | 6.920 | 6.637 | 5.674 | 6.410 |
| HopeNet (α = 2) | 6.470 | 6.559 | 5.436 | 6.155 |
| HopeNet Robust | 9.1709 | 7.0361 | 5.7762 | 7.3277 |
| PoseNet (Robust) | 6.9731 | 6.8398 | 5.8275 | **6.5468** |
| PoseNet less robust | 5.0021 | 6.8970 | 4.7351 | **5.5447** |
| PoseNet no downsampling | 5.4175 | 6.6848 | 4.8637 | **5.6553** |
| PoseNet no augmentation | 5.4444 | 6.1138 | 5.0474 | **5.5352** |

PoseNet has really good accuracy. It wins HopeNet on AFLW2000 and it is a worthy candidate against more advance models like [HopeNet++](https://arxiv.org/abs/1901.06778) and [FSA-Net](https://github.com/shamangary/FSA-Net/blob/master/0191.pdf).

It also has good accuracy on the second validation set:
| Model | θ |
|---|---|
| HopeNet Robust | 8.0483 |
| PoseNet | 7.2909 |
| PoseNet less robust | 6.4961 |
| PoseNet no downsampling | 6.5887 |
| PoseNet no augmentation | 6.5534 |
| Average of all PoseNets | **6.2530** |

### PoseNet_pnp

PoseNet_pnp has trained on the dataset **300W-LP** but uses the PnP algorithm results that has run on the labeled landmarks, as labels, and tested on **second validation set**.

There are 4 different PoseNet_pnp models, there are 4 runs of the same train protocol.

It also has good accuracy on the second validation set:
| Model | θ |
|---|---|
| Average of all PoseNets_pnp | **2.6564** |

PoseNet_pnp has really good accuracy and wins even Dlib and other landmarks detectors.

### Visual Compare

#### PoseNet vs HopeNet

<img src="https://raw.githubusercontent.com/rhalaly/22928_FinalProj/master/readme/PosenetVsHopenet.png" height="300"/>

HopeNet - Left, PoseNet - Right

#### PoseNet vs AFLW2000 Labels

<img src="https://raw.githubusercontent.com/rhalaly/22928_FinalProj/master/readme/PosenetVsAflw.png" height="300"/>

PoseNet - RGB, AFLW2000 Labels - CMY

#### PoseNet vs Validation Set 2.0 Labels

<img src="https://raw.githubusercontent.com/rhalaly/22928_FinalProj/master/readme/PosenetVsValid2.png" height="300"/>

PoseNet - RGB, Validation 2.0 Labels - CMY

#### PoseNet_pnp vs Validation Set 2.0 Labels

<img src="https://raw.githubusercontent.com/rhalaly/22928_FinalProj/master/readme/PosenetpnpVsValid2.png" height="300"/>

PoseNet_pnp - RGB, Validation 2.0 Labels - CMY

#### PoseNet vs Test Set
Images are [here](https://github.com/rhalaly/22928_FinalProj/tree/master/output/posenet), CSV is [here](https://github.com/rhalaly/22928_FinalProj/blob/master/output/posenet/output.csv).

#### PoseNet_pnp vs Test Set
Images are [here](https://github.com/rhalaly/22928_FinalProj/tree/master/output/posenet_pnp), CSV is [here](https://github.com/rhalaly/22928_FinalProj/blob/master/output/posenet_pnp/output.csv).