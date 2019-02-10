import os
import sys
import torch
import pickle
import subprocess
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

from opt import args
from PIL import Image
from torch.autograd import Variable
from matplotlib.pyplot import imshow

sys.path.append('./')
import decode.dec_config as config
from itertools import product

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

    subjects = config.subjects
    rois = config.rois
        
    print("-------- loading model --------")
    
    net = transnet().cuda()
    net.load_state_dict(torch.load('./reconstruction/model/deconvNN_par.pkl'))

    print("-------- reconstruction --------")

    for sbj, roi in product(subjects, rois):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)

        decode_id = 'decode_' + sbj + '_' + roi

        with open(os.path.join(args.decoded_feat, decode_id+'_pred.pkl'), "rb") as f:
            test_x = pickle.load(f)

        with open(os.path.join(args.decoded_feat, decode_id+'_id.pkl'), "rb") as f:
            test_id = pickle.load(f)

        testdata_x = []

        for i in range(len(test_x)):
            testdata_x.append(test_x[i].reshape(4096))
#           testdata_y.append(train_y[train_id[i]])
    
        print("-------- generate coarse image --------")

        test_img_path = './tmp/test_pix2pix/'

        for i in range (len(testdata_x)):
            img_class = test_id[i][0: 9]
            file_path = os.path.join(test_img_path, img_class)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            
            index = [0]
            index[0] = testdata_x[i]
            x = torch.from_numpy(np.array(index)).float().cuda()
            prediction = net(x).cpu().data
            img = transforms.ToPILImage()(prediction[0]).convert('RGB').resize((128, 128))
            img.save(os.path.join(file_path, test_id[i]))
        
#           xxx = torch.from_numpy(testdata_y[i][0])
#           img = transforms.ToPILImage()(xxx).convert('RGB').resize((128, 128))
#           img.save(file_path + test_id[i])

        print("-------- refining image --------")

        output_dir = args.output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range (len(testdata_x)):
            img_class = test_id[i][0: 9]
            img_path = os.path.join(test_img_path, img_class)
            if not os.path.exists(img_path):
                print('image class %s not exist in dataset, skipping' % img_class)
                continue
            
            command = "python ./reconstruction/pix2pix/test.py --dataroot "+ img_path + " --name " + img_class + " --model test --netG unet_128 --direction BtoA --dataset_mode single --norm batch --load_size 128 --crop_size 128 --checkpoints_dir ./reconstruction/model/checkpoints --results_dir " + output_dir
            subprocess.call(command.replace('\n',''), shell=True)
    
    print("-------- reconstruction complete --------")