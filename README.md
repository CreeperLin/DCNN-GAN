# DCNN-GAN

Code repository for MVA 2019 paper "DCNN-GAN: Reconstructing Realistic Image from fMRI"

## Prerequisites

- Linux / macOS
- NVIDIA GPU with CUDA CuDNN
- Python 3
- PyTorch
- scikit-learn

## Getting Started

### Installation

- clone this repo

    git clone https://github.com/CreeperLin/DCNN-GAN.git

- install pytorch (using Anaconda is recommended)

### Image encoding

- Download ILSVRC2012 training set

    ./datasets/download_image.sh

- Compute the image features

    ./encoder/encode.py --datapath ./datasets/image --output-dir ./datasets/feat

### fMRI decoder train/test

- Download fMRI on Imagenet datasets

    ./datasets/download_fmri.sh

- Train fMRI decoder

    ./decode/train.py --datapath ./datasets/fmri

- Test fMRI decoder (output features)

    ./decode/train.py --datapath ./datasets/fmri --output-dir ./datasets/dec_feat

### DCNN-GAN train/test

- Train DCNN-GAN

    python ./dcnn-gan/train.py --datapath ./datasets/feat --name dcnngan --model pix2pix --direction BtoA

- Test DCNN-GAN

    python ./dcnn-gan/test.py --datapath ./datasets/feat

- Image reconstruction using decoded features

    python ./dcnn-gan/test.py --datapath ./datasets/dec_feat

### Run the full pipeline (training & reconstruction)

    ./run_all.sh

## Results

The example reconstructed images are listed below:

## Citation

    @article{Lin2018DCNN-GAN
        author = {Yunfeng, Lin and Jiangbei, Li and Hanjing, Wang",
        title = {DCNN-GAN: Reconstructing Realistic Image from   fMRI},
        year = {2018},
        howpublished={\url{https://github.com/CreeperLin/DCNN-GAN}}
    }

## Acknowledgements

The GAN model is based on the pytorch implementation of pix2pix.

The fMRI data is obtained using the datasets from Generic Object Decoding.
