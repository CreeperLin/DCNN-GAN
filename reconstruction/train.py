import os
import cv2
import torch
import torchvision
import pickle
import subprocess
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.models as models

from opt import args
from PIL import Image
from torch.autograd import Variable
from matplotlib.pyplot import imshow

if __name__=="__main__":
    
    print('-------- loading DCNN training data --------')

    with open(os.path.join(args.DCNN_dataset, 'train_x_vgg19_bn_fc.pickle'), "rb") as f:
        train_x = pickle.load(f)

    with open(os.path.join(args.DCNN_dataset, 'train_y_vgg19_bn_fc.pickle'), "rb") as f:
        train_y = pickle.load(f)

    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
            
    print('-------- training DCNN model --------')
    for i in range(len(train_x)):
        train_x[i].requires_grad = True
        train_y[i].requires_grad = True
    
    traindata = Data.TensorDataset(train_x, train_y)
    
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
    loss_function = nn.MSELoss(reduction='elementwise_mean')
    
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

    torch.save(transNet.state_dict(), './reconstruction/model/deconvNN_par.pkl')
    print('-------- DCNN training complete ----------')
    
    net = models.vgg19_bn(pretrained = True).cuda()
    net.classifier = nn.Sequential(*list(net.classifier.children())[:-6])

    print('-------- generating pix2pix training data ---------')
    img_data = torchvision.datasets.ImageFolder(args.pix2pix_dataset, transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()]))
    testdata = torch.utils.data.DataLoader(img_data)
    pix2pixdata_y = [[]]*len(img_data.classes)
    pix2pixdata_x = [[]]*len(img_data.classes)
    transNet_x = [[]]*len(img_data.classes)
    
    for step, (x, y) in enumerate(testdata):
        b_x = x.cuda()
        data = net(b_x)
        transNet_x[img_data[step][1]].append(data.cpu().data[0].numpy())
        pix2pixdata_y[img_data[step][1]].append((transforms.ToTensor()(transforms.ToPILImage()(x.data[0]).convert('RGB').resize((112, 112), Image.ANTIALIAS))).numpy())
    
    for i in range (len(transNet_x)):
        
        train_img_path = './tmp/train_pix2pix/'

        file_path = os.path.join(train_img_path, img_data.classes[i], 'train')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            
        for j in range (len(transNet_x[i])):
            x = torch.from_numpy(np.array([transNet_x[i][j]])).cuda()
            prediction = transNet(x).cpu().data
            xxx = torch.from_numpy(pix2pixdata_y[i][j])
            yyy = prediction[0]
            z = torch.cat((xxx, yyy), 2)
            img = transforms.ToPILImage()(z).convert('RGB')
            img.save(os.path.join(file_path, str(j) + '.jpg'))
            
    print('-------- pix2pix training data saved ---------')
    
    print('-------- training pix2pix model ----------')
    
    classes = open('./reconstruction/class_list.txt') 
    for line in classes.readlines():
        command = "python ./reconstruction/pix2pix/train.py --dataroot " + os.path.join(train_img_path, line) + " --name " + line + " --model pix2pix --netG unet_128 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --load_size 128 --crop_size 128 --checkpoints_dir ./reconstruction/model/checkpoints --batch_size " + str(args.pix2pix_batch) + " --niter " + str(args.pix2pix_niter) + " --niter_decay " + str(args.pix2pix_niter_decay) + " --lr " + str(args.pix2pix_lr)
        subprocess.call(command.replace('\n',''), shell = True)
        
    print('-------- training complete ----------')