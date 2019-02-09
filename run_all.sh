#!/bin/sh
set -e

# install prerequisites
pip3 install -r requirements.txt

# download fMRI dataset
./datasets/download_fmri.sh

# download image dataset

# train decoder and decode
if [ -d ./tmp/feat_data ]; then
    echo 'feat_data exists, skipping encode'
else
    python3 ./decode/train_dataloader.py --img_data ./datasets/image_fmri --output ./tmp/feat_data
fi
python3 ./decode/decode.py --fmri_data ./datasets/fmri_data --feat_data ./tmp/feat_data --output ./tmp/decoded_feat

# train DCNN-GAN
python3 ./reconstruction/train_dataloader.py --dataset ./datasets/train_dcnn_img --output ./tmp/dcnn_train
python3 ./reconstruction/train.py --DCNN_dataset ./tmp/dcnn_train --pix2pix_dataset ./datasets/train_gan_img

# test DCNN-GAN
python3 ./reconstruction/test.py --decoded_feat ./tmp/decoded_feat --output ./reconstruction/results
