import torch
import pickle
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

from opt import opt
from PIL import Image
from torch.autograd import Variable
from matplotlib.pyplot import imshow

args = opt
if __name__=="__main__":
    class transnet(nn.Module):
        def __init__(self):
            super(transnet, self).__init__()
            self.fc = nn.Sequential(nn.Linear(4096, 512 * 7 * 7))
            self.layer4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride = 2, padding = 1), nn.BatchNorm2d(256), nn.ReLU(True))
            self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride = 2, padding = 1), nn.BatchNorm2d(128), nn.ReLU(True))
            self.layer2 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, stride = 2, padding = 1), nn.BatchNorm2d(128), nn.ReLU(True))
            self.layer1 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, stride = 2, padding = 1), nn.BatchNorm2d(128), nn.ReLU(True))
            self.layer0 = nn.Conv2d(128, 3, 1, stride = 1, padding = 0)
        def forward(self, input):
            x = self.fc(input)
            x = x.reshape(x.shape[0], 512, 7, 7)
            x = self.layer4(x)
            x = self.layer3(x)
            x = self.layer2(x)
            x = self.layer1(x)
            x = self.layer0(x)
            return x
    net = transnet().cuda()
    net.load_state_dict('./model/deconvNN_par.pkl')
    