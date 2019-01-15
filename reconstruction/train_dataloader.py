import torch
import pickle
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision.models as models
import torchvision.transforms as transforms

from opt import opt
from torch.autograd import Variable

args = opt
if __name__=="__main__":
    net = models.vgg19_bn(pretrained = True).cuda()
    net.classifier = nn.Sequential(*list(net.classifier.children())[:-6])

    print('---------loading DCNN training data---------')
    DCNN_data = torchvision.datasets.ImageFolder(args.dataset, transform=transforms.Compose([
                                                    transforms.Scale(256),
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
    
    file1 = open('./data/train_x_vgg19_bn_fc.pickle', 'wb')
    pickle.dump(traindata_x, file1)
    file1.close()
    file2 = open('./data/train_y_vgg19_bn_fc.pickle', 'wb')
    pickle.dump(traindata_y, file2)
    file2.close()

print('---------DCNN training data saved---------')
