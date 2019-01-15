import os
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

    torch.save(transNet.state_dict(), './model/deconvNN_par.pkl')
    print('--------deconv network training complete----------')
    
    net = models.vgg19(pretrained = True).cuda()
    net.classifier = nn.Sequential(*list(net.classifier.children())[:-6])

    print('--------generate pix2pix training data---------')
    img_data = torchvision.datasets.ImageFolder('?/pix2pix_data/train_img', transform=transforms.Compose([
                                                transforms.Scale(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()]))
    testdata = torch.utils.data.DataLoader(img_data)
    pix2pixdata_y = [[]]*len(img_data.classes)
    pix2pixdata_x = [[]]*len(img_data.classes)
    transNet_x = [[]]*len(img_data.classes)
    
    for step, (x, y) in enumerate(testdata):
        b_x = x.cuda()
        data = net(b_x)
        pix2pixdata_y[img_data[step][1]].append(data.cpu().data[0].numpy())
        transNet_x[img_data[step][1]].append((transforms.ToTensor()(transforms.ToPILImage()(x.data[0]).convert('RGB').resize((112, 112), Image.ANTIALIAS))).numpy())
    
    
    for i in range (len(transNet_x)):
        
        file_path = './data/pix2pixdata/' + img_data.classes[i] + '/train/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            
        for j in range (len(transNet_x[i])):
            x = torch.from_numpy(np.array([transNet_x[i][j]])).cuda()
            prediction = transNet(x).cpu().data
            xxx = torch.from_numpy(pix2pixdata_y[i][j])
            yyy = prediction[0]
            z = torch.cat((xxx, yyy), 2)
            img = transforms.ToPILImage()(z),convert('RGB')
            img.save(file_path + str(j) + '.jpg')
            
    print('--------pix2pix training data saved---------')
    
    
    
    