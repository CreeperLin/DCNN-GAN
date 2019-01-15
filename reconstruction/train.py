import cv2
import torch
import pickle
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
    
    print('--------loading training data---------')
    file1 = open('train_x_vgg19_bn_fc.pickle',"rb")
    train_x = pickle.load(file1)
    file1.close()
    file2 = open('train_y_vgg19_bn_fc.pickle',"rb")
    train_y = pickle.load(file2)
    file2.close()
    print('--------loading training data complete---------')
    print('--------training deconv network----------')
    for i in range(len(train_x)):
        train_x[i] = torch.from_numpy(train_x[i])
        train_y[i] = torch.from_numpy(train_y[i])
        train_x[i].requires_grad = True
        train_y[i].requires_grad = True
    
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
    transNet = transnet().cuda()
    
    train_loader = Data.DataLoader(traindata, batch_size = args.DCNN_batch, shuffle = True)
    optimizer = torch.optim.Adam(transNet.parameters(), lr = args.DCNN_lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    loss_function = nn.MSELoss(size_average=True)
    
    for t in range(args.DCNN_epoch):
        for step, (x, y) in enumerate(train_loader):
            b_x = x.cuda()
            b_y = y.cuda()
            prediction = transNet(b_x)
            loss = loss_function(prediction, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
        
            if(step % 100 == 0):
                print("train epoch: ", t, "loss", loss.cpu().data.numpy())

    torch.save(transNet, 'model1.pkl')
    print('--------deconv network training complete----------')
    
    
    