#!/bin/sh

# install prerequisites
pip3 install -r requirements.txt

# download fMRI dataset
./datasets/download_fmri.sh

# download image dataset

# train decoder and decode
python ./decode/train_dataloader.py --img_data ./datasets/image_fmri --output ./tmp/feat_data
python ./decode/decode.py --fmri_data ./datasets/fmri_data --feat_data ./tmp/feat_data --output ./tmp/decoded_feat

# train DCNN-GAN

# test DCNN-GAN

# reconstruct the image
