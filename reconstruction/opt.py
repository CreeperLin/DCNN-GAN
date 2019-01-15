import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch DCNN-GAN training')

"--------------dataloader options------------"
parser.add_argument('--dataset', default='../datasets/Imagenet2012/img_par', type=str, help='path to images')


"--------------train options------------"
parser.add_argument('--DCNN_lr', default=2e-4, type=float, help='learning rate of DCNN')

parser.add_argument('--DCNN_batch', default=200, type=int, help='batch size of DCNN')

parser.add_argument('--DCNN_epoch', default=200, type=int, help='training epoch of DCNN')

parser.add_argument('--pix2pix_dataset', default='../datasets/pix2pix_data/train_img', type=int, help='path to pix2pix training data')

parser.add_argument('--pix2pix_lr', default=0.0002, type=float, help='learning rate of pix2pix')

parser.add_argument('--pix2pix_niter', default=100, type=int, help='pix2pix iter at starting learning rate')

parser.add_argument('--pix2pix_niter_decay', default=100, type=int, help='pix2pix iter to linearly decay learning rate to zero')

parser.add_argument('--pix2pix_batch', default=1, type=int, help='pix2pix batch size')

"--------------test options-------------"

parser.add_argument('--decoded_feat', default='./tmp/decoded_feat', type=str, help='decoded features file')