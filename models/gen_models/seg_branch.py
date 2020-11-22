# -*- coding: utf-8 -*-
# @Time    : 19/11/5 9:39
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : .py
# Returns 2D convolutional layer with space-preserving padding
from torch import nn
import torch.nn.functional as F
import torch
from torch.nn import init
import numpy as np
from module.aspp import SegASPPDecoder, ASPP

class DeepLabDecoder(nn.Module):

    def __init__(self, backbone='vgg16bn', num_class=1, output_stride=16, return_features=False):
        super(DeepLabDecoder, self).__init__()
        batchnorm = nn.BatchNorm2d
        self.backbone = backbone
        self.aspp = ASPP(backbone, output_stride, batchnorm)
        self.decoder = SegASPPDecoder(num_class, backbone)
        self.return_features = return_features
        self.noisy_features = False

    def set_return_features(self, return_features):
        self.return_features = return_features

    def set_noisy_features(self, noisy_features):
        self.noisy_features = noisy_features

    def forward(self, input):
        if self.noisy_features is True:
            noise_input = np.random.normal(loc=0.0, scale=abs(input.mean().cpu().item() * 0.05),
                                           size=input.shape).astype(np.float32)
            input = input + torch.from_numpy(noise_input).cuda()

        if 'vgg' in self.backbone:
            x, low_level_feat = input.relu5, input.relu3
        elif 'resnet' in self.backbone:
            x, low_level_feat = input.layer4, input.layer1
        else:
            raise Exception('Unknown backbone')

        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            noise_low_level_feat = np.random.normal(loc=0.0, scale=abs(low_level_feat.mean().cpu().item() *
                                                                       0.5), size=low_level_feat.shape).astype(
                np.float32)
            x += torch.from_numpy(noise_x).cuda()
            low_level_feat += torch.from_numpy(noise_low_level_feat).cuda()

        x = self.aspp(x)
        aspp = x
        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            x += torch.from_numpy(noise_x).cuda()

        low_res_x, features = self.decoder(x, low_level_feat)
        x = F.interpolate(low_res_x, scale_factor=4, mode='bilinear', align_corners=True)
        if self.return_features:
            return x, features, aspp
        return x

from module.Unet_parts_2d import DecoderBlock
from torch.nn.parameter import Parameter

class UnetEntangleDecoderFusion(nn.Module):
    def __init__(
            self,
            backbone='vgg16bn',
            filter_num=32,
            num_class=1,
            attention_type='scse'
    ):
        super().__init__()
        use_batchnorm = True
        self.bb = backbone
        filter_list_2 = [32, 16, 8, 2, 1]
        filter_list_1 = [32 + 64, filter_list_2[0] + 16, filter_list_2[1] + 8, filter_list_2[2] + 2, 2, 1]
        self.layer1 = DecoderBlock(filter_num * filter_list_1[0], filter_num * filter_list_2[0],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer2 = DecoderBlock(filter_num * filter_list_1[1], filter_num * filter_list_2[1],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer3 = DecoderBlock(filter_num * filter_list_1[2], filter_num * filter_list_2[2],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer4 = DecoderBlock(filter_num * filter_list_1[3], filter_num * filter_list_2[3],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer5 = DecoderBlock(filter_num * filter_list_1[4], filter_num * filter_list_2[4],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.final_conv = nn.Conv2d(filter_num * filter_list_1[5], num_class, kernel_size=(1, 1))
        self.gamma1 = Parameter(torch.zeros(1))
        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(1 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.gamma2 = Parameter(torch.zeros(1))
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(1 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.gamma3 = Parameter(torch.zeros(1))
        self.fusion_conv3 = nn.Sequential(
            nn.Conv2d(1 + 1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.gamma4 = Parameter(torch.zeros(1))
        self.fusion_conv4 = nn.Sequential(
            nn.Conv2d(1 + 2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        # self.fusion_conv1 = nn.Sequential(
        #     nn.Conv2d(1 + 256, 256, kernel_size=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )
        # self.fusion_conv2 = nn.Sequential(
        #     nn.Conv2d(1 + 512, 512, kernel_size=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )
        # self.fusion_conv3 = nn.Sequential(
        #     nn.Conv2d(1 + 1024, 1024, kernel_size=1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(inplace=True)
        # )
        # self.fusion_conv4 = nn.Sequential(
        #     nn.Conv2d(1 + 2048, 2048, kernel_size=1),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(inplace=True)
        # )
        self.apply(weights_init_normal)

    def forward(self, backbone_features, fusion, in_feature):
        if fusion is None:
            x0, x1, x2, x3, x4 = backbone_features.layer0, backbone_features.layer1, backbone_features.layer2, \
                                 backbone_features.layer3, backbone_features.layer4
        else:
            x0, x1, x2, x3, x4 = backbone_features.layer0, backbone_features.layer1, backbone_features.layer2, \
                                 backbone_features.layer3, fusion

        x1 = self.fusion_conv1(torch.cat([x1, self.gamma1 * torch.sum(in_feature[-4], dim=1).unsqueeze(1)], dim=1))
        x2 = self.fusion_conv2(torch.cat([x2, self.gamma2 * torch.sum(in_feature[-3], dim=1).unsqueeze(1)], dim=1))
        x3 = self.fusion_conv3(torch.cat([x3, self.gamma3 * torch.sum(in_feature[-2], dim=1).unsqueeze(1)], dim=1))
        x4 = self.fusion_conv4(torch.cat([x4, self.gamma4 * torch.sum(in_feature[-1], dim=1).unsqueeze(1)], dim=1))
        # x1 = self.fusion_conv1(torch.cat([x1, torch.sum(in_feature[-4], dim=1).unsqueeze(1)], dim=1))
        # x2 = self.fusion_conv2(torch.cat([x2, torch.sum(in_feature[-3], dim=1).unsqueeze(1)], dim=1))
        # x3 = self.fusion_conv3(torch.cat([x3, torch.sum(in_feature[-2], dim=1).unsqueeze(1)], dim=1))
        # x4 = self.fusion_conv4(torch.cat([x4, torch.sum(in_feature[-1], dim=1).unsqueeze(1)], dim=1))
        x = self.layer1([x4, x3])
        x = self.layer2([x, x2])
        x = self.layer3([x, x1])
        x = self.layer4([x, x0])
        x = self.layer5([x, None])
        x = self.final_conv(x)
        return x

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
