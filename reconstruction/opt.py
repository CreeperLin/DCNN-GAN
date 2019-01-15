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

"--------------test options-------------"

parser.add_argument('--test_feat', default='../decode/result/decode_Subject1_VC_lr_pred.pkl', type=str, help='decoded features file')

parser.add_argument('--test_id', default='../decode/result/decode_Subject1_VC_lr_id.pkl', type=str, help='path to decoded images id file')
