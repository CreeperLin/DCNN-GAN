#!/bin/sh

# install prerequisites
pip3 install -r requirements.txt

# download fMRI dataset
./datasets/download_fmri.sh

# download image dataset

# train decoder and decode
python3 ./decode/train_dataloader.py --img_data ./datasets/image_fmri --output ./tmp/feat_data
python3 ./decode/decode.py --fmri_data ./datasets/fmri_data --feat_data ./tmp/feat_data --output ./tmp/decoded_feat

# train DCNN-GAN
python3 ./reconstruction/dataloader.py --dataset ./datasets/Imagenet2012/img_par
python3 ./reconstruction/train.py --pix2pix_dataset ./datasets/pix2pix_data/train_img

# test DCNN-GAN
python3 ./reconstruction/test.py --decoded_feat ./tmp/decoded_feat
