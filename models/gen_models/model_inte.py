# -*- coding: utf-8 -*-
# @Time    : 19/11/4 20:45
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    :
import torch
import torch.nn as nn
from models.gen_models.seg_branch import UnetEntangleDecoderFusion
from module.gen_backbone import BACKBONE
from collections import namedtuple
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
from module.attention_module import *
from collections import OrderedDict


class Integrate_Model_Cls_Ensemble_CAM_Att(nn.Module):
    def __init__(self, backbone='resnet101', num_classes=3, noisy_features=False):
        super(Integrate_Model_Cls_Ensemble_CAM_Att, self).__init__()
        self.noisy_features = noisy_features
        filter_num = 32
        # --------cls-----------
        resnet = models.resnet101(pretrained=True)  # pretrained ImageNet
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        resnet_cov1_weight = resnet.conv1.weight.data
        nv = torch.cat([resnet_cov1_weight, torch.unsqueeze(torch.mean(resnet_cov1_weight, dim=1), dim=1)], dim=1)
        self.conv1.weight.data.copy_(nv)
        # torch.nn.init.normal_(self.conv1.weight.data, 0.0, 0.02)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.cls_branch = nn.Sequential(
            nn.Linear(filter_num * 64, num_classes),
        )
        self.ca = CAM_Module()

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.conv1)
        small_lr_layers.append(self.layer1)
        small_lr_layers.append(self.layer2)
        small_lr_layers.append(self.layer3)
        small_lr_layers.append(self.layer4)
        return small_lr_layers

    def get_backbone_out(self, image, seg_out, seg_b_out):
        layer0 = self.relu(self.bn1(self.conv1(torch.cat([image, seg_out], dim=1))))
        x = self.max_pool(layer0)
        seg_b_out_0 = self.max_pool(seg_b_out[0])
        # layer1 = self.layer1(x * (1 + torch.sigmoid(seg_b_out_0)))
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1 * (1 + torch.sigmoid(seg_b_out[1])))
        layer3 = self.layer3(layer2 * (1 + torch.sigmoid(seg_b_out[2])))
        layer4 = self.layer4(layer3 * (1 + torch.sigmoid(seg_b_out[3])))
        return (layer1, layer2, layer3, layer4)

    def get_last_layer_out(self, image, seg_out, seg_b_out):
        layer0 = self.relu(self.bn1(self.conv1(torch.cat([image, seg_out], dim=1))))
        x = self.max_pool(layer0)
        seg_b_out_0 = self.max_pool(seg_b_out[0])
        # layer1 = self.layer1(x * (1 + torch.sigmoid(seg_b_out_0)))
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1 * (1 + torch.sigmoid(seg_b_out[1])))
        layer3 = self.layer3(layer2 * (1 + torch.sigmoid(seg_b_out[2])))
        layer4 = self.layer4(layer3 * (1 + torch.sigmoid(seg_b_out[3])))
        layer4 = layer4 * (1 + torch.sigmoid(seg_b_out[4]))
        cls_back_out = self.ca(layer4)
        x = F.adaptive_avg_pool2d(cls_back_out, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, image, seg_out, seg_b_out, branch='cls'):
        layer0 = self.relu(self.bn1(self.conv1(torch.cat([image, seg_out], dim=1))))
        x = self.max_pool(layer0)
        seg_b_out_0 = self.max_pool(seg_b_out[0])
        # layer1 = self.layer1(x * (1 + torch.sigmoid(seg_b_out_0)))
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1 * (1 + torch.sigmoid(seg_b_out[1])))
        layer3 = self.layer3(layer2 * (1 + torch.sigmoid(seg_b_out[2])))
        layer4 = self.layer4(layer3 * (1 + torch.sigmoid(seg_b_out[3])))
        layer4 = layer4 * (1 + torch.sigmoid(seg_b_out[4]))
        cls_back_out = self.ca(layer4)
        x = F.adaptive_avg_pool2d(cls_back_out, (1, 1))
        x = x.view(x.size(0), -1)
        cls_out = self.cls_branch(x)
        return cls_out


class Integrate_Model_Seg_Ensemble_Fusion(nn.Module):
    def __init__(self, backbone='resnet101', n_channels=3, num_classes=1, noisy_features=False):
        super(Integrate_Model_Seg_Ensemble_Fusion, self).__init__()
        self.noisy_features = noisy_features
        self.backbone = BACKBONE[backbone](backbone=backbone, pretrained=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dua = DANetHead(2048, 2048)
        self.seg_branch = UnetEntangleDecoderFusion(backbone=backbone, num_class=num_classes)

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.backbone)
        return small_lr_layers

    def forward(self, image, cam=None, cls_b_out=None, dua=False):
        backbone_out = self.backbone(image)
        fusion = backbone_out[-1]
        if dua:
            fusion = self.dua(fusion)
        out = self.seg_branch(backbone_out, fusion, cls_b_out)
        return out

