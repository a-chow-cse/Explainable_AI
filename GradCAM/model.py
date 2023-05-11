import math
import timm
import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl

# from iin_models.ae import IIN_AE, IIN_RESNET_AE

class ResNet50(nn.Module):
    def __init__(self, pretrain=True, num_classes=10, img_ch=1):
        super().__init__()
        weights = None

        if pretrain:
            #weights = ResNet50_Weights.IMAGENET1K_V2
            model_resnet = models.resnet50(pretrained=True)
        else:
            model_resnet = models.resnet50(weights=weights)
        if img_ch != 3:
            self.conv1 = nn.Conv2d(img_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.linear = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        x = self.get_features(x)
        x = self.linear(x)

        return x
    
    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ImageClassifier(nn.Module):
    def __init__(self, class_num=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear = nn.Linear(20*7*7, class_num)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
        return out





class ViTBase(pl.LightningModule):
    def __init__(self, num_classes, img_ch=1):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)

        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)


    def forward(self, x):
        return self.backbone(x)