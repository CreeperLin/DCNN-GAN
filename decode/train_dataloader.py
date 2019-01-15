import torch
import pickle
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable

net = models.vgg19(pretrained = True).cuda()
net.classifier = nn.Sequential(*list(net.classifier.children())[:-6])

print('---------loading decode training data---------')
img_data = torchvision.datasets.ImageFolder('?', transform=transforms.Compose([
                                                transforms.Scale(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()]))
testdata = torch.utils.data.DataLoader(img_data)
features = {}
image = {}

for step, (x, y) in enumerate(testdata):
    b_x = x.cuda()
    data = net(b_x)
    xxx = data.cpu().data.numpy()
    features[os.path.basename(img_data.imgs[step][0])] = xxx.flatten()
    image[os.path.basename(img_data.imgs[step][0])] = x.numpy()

file1 = open('images_vgg19_bn_fc.pickle', 'wb')
pickle.dump(image, file1)
file1.close()

file2 = open('features_vgg19_bn_fc.pickle', 'wb')
pickle.dump(features, file2)
file2.close()
print('---------decode training data saved---------')