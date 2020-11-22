from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import glob as gb
import torch
import numpy as np
import random
import warnings
from scipy import ndimage
import cv2
from driver import std, mean
from os.path import join
from commons.utils import visualize

transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.1),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.1),
    transforms.ColorJitter(hue=0.15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])


def rescale_crop(image, scale, num, mode, output_size=224):
    image_list = []
    #TODO  for cls, it should be h,w.
    #TODO but for seg, it could be w,h
    h, w = image.size
    if mode == "test":
        trans = transforms.Compose([
            transforms.CenterCrop((int(h * scale) + 500 * scale, int(w * scale) + 500 * scale)),
            transforms.RandomCrop((int(h * scale), int(w * scale))),
            # transforms.CenterCrop((int(h * scale) + 500 * scale, int(w * scale) + 500 * scale)),
            # transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
            #                         scale=[0.7, 1.3]),
            # transforms.RandomRotation((-10, 10)),
            # transforms.RandomCrop((int(h * scale), int(w * scale))),

            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((output_size, output_size)),
        ])
    elif mode == "train":
        trans = transforms.Compose([
            transforms.CenterCrop((int(h * scale) + 500 * scale, int(w * scale) + 500 * scale)),
            transforms.RandomCrop((int(h * scale), int(w * scale))),
            # transforms.RandomCrop((int(h * scale), int(w * scale))),
            transforms.RandomAffine([-90, 90], translate=[0.01, 0.1],
                                    scale=[0.9, 1.1]),
            transforms.RandomRotation((-10, 10)),
            transforms.Resize((output_size, output_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
    elif mode == "train_seg":
        trans = transforms.Compose([
            # transforms.RandomCrop((int(h * scale), int(w * scale))),
            transforms.RandomAffine(0, translate=[0.01, 0.1],
                                    scale=[0.9, 1.1], shear=[-10, 10]),
            transforms.CenterCrop((int(h * scale) + 500 * scale, int(w * scale) + 500 * scale)),
            transforms.RandomRotation((-180, 180)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((output_size, output_size)),
            # transforms.RandomChoice([
            #     transforms.ColorJitter(brightness=0.1),
            #     transforms.ColorJitter(contrast=0.2),
            #     transforms.ColorJitter(saturation=0.1),
            #     transforms.ColorJitter(hue=0.15),
            #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            #     transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
            #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            # ])
        ])
    elif mode == "test_seg":
        trans = transforms.Compose([
            transforms.Resize(output_size),
        ])
    for i in range(num):
        img = trans(image)
        image_list.append(img)
    return image_list


def rescale_crop_test(image, num, output_size=(244, 244)):
    image_list = []
    trans = transforms.Compose([
        transforms.Resize(output_size),
    ])
    for i in range(num):
        img = trans(image)
        image_list.append(img)
    return image_list


def crop(image, mode, output_size=224):
    image_list = []
    if mode == "train":
        trans = transforms.Compose([
            transforms.RandomRotation((-10, 10)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((output_size, output_size)),  # change the order
        ])
    if mode == "val":
        trans = transforms.Compose([
            transforms.Resize((output_size, output_size)),
        ])
    if mode == "normal":
        trans = transforms.Compose([
            transforms.RandomAffine([-90, 90], translate=[0.01, 0.1],
                                    scale=[0.9, 1.1]),
            transforms.RandomRotation((-10, 10)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((output_size, output_size)),
        ])
    img = trans(image)
    image_list.append(img)
    return image_list


class argumentation_train_seg_aug(object):
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, image, groundtruth, mode='train_seg'):
        image_list = []
        gt_list = []
        w, h = image.size
        for ii in range(3):
            scale = np.random.rand() + 0.2
            seed = np.random.randint(2147483647)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            image_list0_img = rescale_crop(image, scale, 1, mode, output_size=self.output_size)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            image_list0_gt = rescale_crop(groundtruth, scale, 1, mode, output_size=self.output_size)
            image_list += image_list0_img
            gt_list += image_list0_gt
            # image_list0_img[0].save(join('./tmp', "image_list_" + str(ii) + "_img_images.png"))
            # image_list0_gt[0].save(join('./tmp', "image_list_" + str(ii) + "_seg_img_images.png"))
        # image.save(join('./tmp', "image_list_raw_img_images.png"))
        # groundtruth.save(join('./tmp', "image_list_seg_img_images.png"))
        image_list += crop(image, "val", output_size=self.output_size)
        gt_list += crop(groundtruth, "val", output_size=self.output_size)

        nomalize_img = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                             std=std)])(crop) for crop
             in
             crops]))
        nomalize_gt = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor()])(crop) for crop
             in
             crops]))
        image_list = nomalize_img(image_list)
        gt_list = nomalize_gt(gt_list)
        gt_list[gt_list > 0] = 1
        # print(torch.unique(gt_list))
        # for i in range(len(image_list)):
        #     visualize(np.transpose(np.array(image_list[i]), (1, 2, 0)) * std + mean,
        #               join('./tmp', "image_list_" + str(i) + "_img_images"))
        #     visualize(np.transpose(np.array(gt_list[i]), (1, 2, 0)),
        #               join('./tmp', "image_list_" + str(i) + "_gt_images"))
        # image.save(join('./tmp', "image_list_raw_img_images.png"))
        # groundtruth.save(join('./tmp', "image_list_seg_img_images.png"))
        return image_list, gt_list


class argumentation_train(object):
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, image, groundtruth, mode='train'):
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list1_img = rescale_crop(image, 0.25, 1, mode, output_size=self.output_size)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list1_gt = rescale_crop(groundtruth, 0.25, 1, mode, output_size=self.output_size)
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list2_img = rescale_crop(image, 0.5, 1, mode, output_size=self.output_size)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list2_gt = rescale_crop(groundtruth, 0.5, 1, mode, output_size=self.output_size)
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list3_img = rescale_crop(image, 0.75, 1, mode, output_size=self.output_size)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list3_gt = rescale_crop(groundtruth, 0.75, 1, mode, output_size=self.output_size)
        image_list_img = crop(image, "val", output_size=self.output_size)
        image_list_gt = crop(groundtruth, "val", output_size=self.output_size)
        image_list = image_list1_img + image_list2_img + image_list3_img + image_list_img
        gt_list = image_list1_gt + image_list2_gt + image_list3_gt + image_list_gt
        nomalize_img = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                             std=std)])(crop) for crop
             in
             crops]))
        nomalize_gt = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor()])(crop) for crop
             in
             crops]))
        image_list = nomalize_img(image_list)
        gt_list = nomalize_gt(gt_list)
        gt_list[gt_list > 0] = 1
        # gt_list[gt_list <= 0.2] = 0
        # for i in range(len(image_list)):
        #     visualize(np.transpose(np.array(image_list[i]), (1, 2, 0)) * std + mean,
        #               join('./tmp', "image_list_" + str(i) + "_img_images"))
        #     visualize(np.transpose(np.array(gt_list[i]), (1, 2, 0)),
        #               join('./tmp', "image_list_" + str(i) + "_gt_images"))
        # image.save(join('./tmp', "image_list_raw_img_images.png"))
        # groundtruth.save(join('./tmp', "image_list_seg_img_images.png"))
        return image_list, gt_list


class argumentation_train_normal(object):
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, image, groundtruth, mode='train'):
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list_img = crop(image, "val", output_size=self.output_size)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list_gt = crop(groundtruth, "val", output_size=self.output_size)
        image_list = image_list_img
        gt_list = image_list_gt
        nomalize_img = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                             std=std)])(crop) for crop
             in
             crops]))
        nomalize_gt = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor()])(crop) for crop
             in
             crops]))
        image_list = nomalize_img(image_list)
        gt_list = nomalize_gt(gt_list)
        gt_list[gt_list > 0] = 1
        # gt_list[gt_list <= 0.2] = 0
        # for i in range(len(image_list)):
        #     visualize(np.transpose(np.array(image_list[i]), (1, 2, 0)) * std + mean,
        #               join('./tmp', "image_list_" + str(i) + "_img_images"))
        #     visualize(np.transpose(np.array(gt_list[i]), (1, 2, 0)),
        #               join('./tmp', "image_list_" + str(i) + "_gt_images"))
        return image_list, gt_list


class argumentation_val_normal(object):
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, image, groundtruth, mode='test'):
        image_list_img = crop(image, "val", output_size=self.output_size)
        image_list_gt = crop(groundtruth, "val", output_size=self.output_size)
        image_list = image_list_img
        gt_list = image_list_gt
        nomalize_img = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                             std=std)])(crop) for crop
             in
             crops]))
        nomalize_gt = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor()])(crop) for crop
             in
             crops]))
        image_list = nomalize_img(image_list)
        gt_list = nomalize_gt(gt_list)
        return image_list, gt_list


class argumentation_val(object):
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, image, groundtruth, mode='test'):
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list1_img = rescale_crop(image, 0.2, 1, mode, output_size=self.output_size)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list1_gt = rescale_crop(groundtruth, 0.2, 1, mode, output_size=self.output_size)
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list2_img = rescale_crop(image, 0.4, 1, mode, output_size=self.output_size)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list2_gt = rescale_crop(groundtruth, 0.4, 1, mode, output_size=self.output_size)
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list3_img = rescale_crop(image, 0.6, 1, mode, output_size=self.output_size)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list3_gt = rescale_crop(groundtruth, 0.6, 1, mode, output_size=self.output_size)
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list4_img = rescale_crop(image, 0.8, 1, mode, output_size=self.output_size)
        random.seed(seed)
        torch.manual_seed(seed)
        image_list4_gt = rescale_crop(groundtruth, 0.8, 1, mode, output_size=self.output_size)

        image_list_img = crop(image, "val", output_size=self.output_size)
        image_list_gt = crop(groundtruth, "val", output_size=self.output_size)
        image_list = image_list1_img + image_list2_img + image_list3_img + image_list4_img + image_list_img
        gt_list = image_list1_gt + image_list2_gt + image_list3_gt + image_list4_gt + image_list_gt
        nomalize_img = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                             std=std)])(crop) for crop
             in
             crops]))
        nomalize_gt = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor()])(crop) for crop
             in
             crops]))
        image_list = nomalize_img(image_list)
        gt_list = nomalize_gt(gt_list)
        return image_list, gt_list


class argumentation_test(object):
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, image, mode='test'):
        image_list1 = rescale_crop(image, 0.2, 1, mode, output_size=self.output_size)
        image_list2 = rescale_crop(image, 0.4, 1, mode, output_size=self.output_size)
        image_list3 = rescale_crop(image, 0.6, 1, mode, output_size=self.output_size)
        image_list4 = rescale_crop(image, 0.8, 1, mode, output_size=self.output_size)
        image_list5 = crop(image, "val", output_size=self.output_size)
        image_list = image_list1 + image_list2 + image_list3 + image_list4 + image_list5
        # image_list =  image_list5
        nomalize = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                             std=std)])(crop) for crop
             in
             crops]))
        image_list = nomalize(image_list)
        return image_list
