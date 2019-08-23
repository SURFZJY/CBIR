import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable 
from PIL import Image
import os
import numpy as np
import pickle


dataset_dir = '/home1/surfzjy/data/holiday'
fileList = os.listdir(dataset_dir)

feat_dict = {}

resnet = models.resnet18(pretrained=True)
modules = list(resnet.children())[:-1]      # delete the last fc layer.
model = nn.Sequential(*modules)

model.eval()  ## unable BatchNormalization and Dropout
model.cuda()

TARGET_IMG_SIZE = 224

for i in range(len(fileList)):

    img_id = fileList[i].split('.')[0]
    print(i, img_id)

    imgpath = os.path.join(dataset_dir, fileList[i])
    img = Image.open(imgpath)
    img = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor = transforms.ToTensor()(img)
    tensor = tensor.resize_(1, 3, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
    tensor = tensor.cuda()

    result = model(Variable(tensor))
    result_npy = result.data.cpu().numpy()  ## (1, 512, 1, 1)
    feat = np.squeeze(result_npy)
    feat_dict[img_id] = feat

# print(feat_dict)
with open("holiday_resnet18_feat.pk", "wb") as f:
    pickle.dump(feat_dict, f)
# print(feat.shape)
# print(type(feat))

# print(tensor.shape)
# print(resnet18._modules.keys())
# reference: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
'''reference: 
https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
https://discuss.pytorch.org/t/resnet-features-not-work/3423/2
'''