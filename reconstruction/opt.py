import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch DCNN-GAN training')

"--------------train options------------"
parser.add_argument('--DCNN_lr', default=2e-4, type=float, help='learning rate of DCNN')

parser.add_argument('--DCNN_batch', default=200, type=int, help='batch size of DCNN')

parser.add_argument('--DCNN_epoch', default=200, type=int, help='training epoch of DCNN')


"--------------test options-------------"


