import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.functional import conv2d

from utils import *
from evaluation import CausalMetric, auc, gkern
from explanations import RISE
from model import ImageClassifier, ResNet50, ViTBase
import matplotlib.cm as cm

cudnn.benchmark = True

# Load black box model for explanations
model = ResNet50(num_classes=200, img_ch=3)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load('../resnet_temp.pt', map_location=device))
model = model.eval()
model = model.cuda()

for p in model.parameters():
    p.requires_grad = False
    
# To use multiple GPUs
model = nn.DataParallel(model)

klen = 11
ksig = 5
kern = gkern(klen, ksig)

# Function that blurs input image
blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

plt.figure(figsize=(12, 4))
img = read_tensor('../cub/val2017/3.jpg') #.................

plt.axis('off')
plt.imshow(kern[0, 0])
plt.savefig("output/Blank.jpg")

plt.axis('off')
tensor_imshow(blur(img)[0])
plt.savefig("output/blurred.jpg")

insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)

explainer = RISE(model, (224, 224))
explainer.generate_masks(N=5000, s=10, p1=0.1)

out = model(img)
_, preds = torch.max(out, dim=1)



class_number=preds[0].item()
print("Class: ",class_number) #...........................
#class_number=
sal = explainer(img.cuda())[class_number].cpu().numpy()
# same image, diff query
#same class, same query, diff image

tensor_imshow(img[0])
plt.axis('off')
plt.imshow(sal, cmap='jet', alpha=0.5)
# Get the 'jet' colormap and reverse it
jet_reversed = cm.get_cmap('jet').reversed()

# Set the reversed colormap
plt.set_cmap(jet_reversed)
plt.savefig("output/rise_hm.jpg")

h = deletion.single_run(img, sal, verbose=1,save_to="./output/del.jpg")
h = insertion.single_run(img, sal, verbose=1,save_to="./output/ins.jpg")