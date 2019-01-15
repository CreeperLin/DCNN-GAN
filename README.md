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

- Clone this repo

```bash
git clone https://github.com/CreeperLin/DCNN-GAN.git
cd DCNN-GAN
```

- Install pytorch (using Anaconda is recommended)

### Image encoding

- Download ILSVRC2012 training set

```bash
./datasets/download_image.sh
```

- Compute the image features

```bash
python ./encoder/encode.py --dataset ./datasets/image --output ./datasets/feat
```

### fMRI decoder train/test

- Download fMRI on Imagenet datasets

```bash
./datasets/download_fmri.sh
```

- Train fMRI decoder and decode

```bash
python ./decode/decode.py --fmri_data ./datasets/fmri_data --feat_data ./datasets/feat_data --output ./tmp/decoded_feat
```

### DCNN-GAN train/test
- Data Preparation

```bash
python ./reconstruction/dataloader.py --dataset ./datasets/Imagenet2012/img_par
```

- Train DCNN-GAN

```bash
python ./reconstruction/train.py --pix2pix_dataset ./datasets/pix2pix_data/train_img
```

- Test DCNN-GAN

```bash
python ./reconstruction/test.py --test_feat ./decode/result/decode_Subject1_VC_lr_pred.pkl --test_id ./decode/result/decode_Subject1_VC_lr_id.pkl
```

### Run the full pipeline (training & reconstruction)

```bash
./run_all.sh
```

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
