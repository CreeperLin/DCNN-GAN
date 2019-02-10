import os
import torch
import pickle
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision.models as models
import torchvision.transforms as transforms

from opt import args
from torch.autograd import Variable
from PIL import Image

if __name__=="__main__":
    net = models.vgg19_bn(pretrained = True).cuda()
    net.classifier = nn.Sequential(*list(net.classifier.children())[:-6])

    print('-------- loading DCNN training images ---------')
    DCNN_data = torchvision.datasets.ImageFolder(args.dataset, transform=transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor()]))
    testdata = torch.utils.data.DataLoader(DCNN_data)
    traindata_x = []
    traindata_y = []

    for step, (x, y) in enumerate(testdata):
        b_x = x.cuda()
        data = net(b_x)
        traindata_x.append(data.cpu().data.numpy()[0])
        traindata_y.append(x.data.numpy()[0])

    for i in range(len(traindata_x)):
        traindata_y[i] = ((transforms.ToTensor()(transforms.ToPILImage()(torch.from_numpy(traindata_y[i])).convert('RGB').resize((112, 112), Image.ANTIALIAS))).numpy())

    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    traindata_x = np.array(traindata_x)
    traindata_y = np.array(traindata_y)
    
    with open(os.path.join(output_dir, 'train_x_vgg19_bn_fc.pickle'), 'wb') as f:
        pickle.dump(traindata_x, f)

    with open(os.path.join(output_dir, 'train_y_vgg19_bn_fc.pickle'), 'wb') as f:
        pickle.dump(traindata_y, f)

print('-------- DCNN training data saved ---------')
