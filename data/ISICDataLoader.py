# -*- coding: utf-8 -*-
# @Time    : 19/11/7 10:01
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : InstanceDataLoader.py
from torch.utils import data
import numpy as np
import os
from os.path import isdir
from data import *
from commons.utils import visualize
from os.path import join
from PIL import Image
import random
import torch
import pandas as pd
from data import ISIC2017, ISIC2018
from driver import train_transforms
from glob import glob

TMP_DIR = "./tmp"
if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

DATATYPES = ['labelled', 'valid', 'test', 'unlabeled']


def make_seg_train_dataset(dir='../../../medical_data/ISIC_2017_Skin_Lesion/'):
    img_dir = os.path.join(dir, "ISIC_train_data")
    gt_dir = os.path.join(dir, "ISIC_train_gt")
    image_paths = []
    image_gt = []
    image_label = []
    image_name = []
    img_filelist = os.listdir(img_dir)
    for index in range(len(img_filelist)):
        img_name = img_filelist[index].rsplit('.', maxsplit=1)[0]
        gt_name = img_name + '_segmentation.png'
        image_paths.append(join(img_dir, img_filelist[index]))
        image_gt.append(join(gt_dir, gt_name))
        image_label.append(1)
        image_name.append(img_name)
    return image_paths, image_gt, image_label, image_name


def make_2017_dataset(dir='../../../medical_data/ISIC_2017_Skin_Lesion/', mode='training'):
    if mode == "training":
        img_dir = os.path.join(dir, "ISIC-2017_Training_Data")
        gt_dir = os.path.join(dir, "ISIC-2017_Training_Part1_GroundTruth")
        csv_filename = os.path.join(dir, "ISIC-2017_Training_Part3_GroundTruth.csv")
    elif mode == "validation":
        img_dir = os.path.join(dir, "ISIC-2017_Validation_Data")
        gt_dir = os.path.join(dir, "ISIC-2017_Validation_Part1_GroundTruth")
        csv_filename = os.path.join(dir, "ISIC-2017_Validation_Part3_GroundTruth.csv")
    else:
        raise ValueError('mode error')
    label_list = pd.read_csv(csv_filename)
    image_paths = []
    image_gt = []
    image_label = []
    image_name = []
    for index, row in label_list.iterrows():
        img_path = os.path.join(img_dir, row["image_id"] + ".jpg")
        gt_path = os.path.join(gt_dir, row["image_id"] + "_segmentation.png")
        img_name = img_path.split("/")[-1]
        label = np.array(
            [row["melanoma"], row["seborrheic_keratosis"], 1 - row["melanoma"] - row["seborrheic_keratosis"]])
        label = np.argmax(label)
        image_paths.append(img_path)
        image_gt.append(gt_path)
        image_label.append(label)
        image_name.append(img_name)
    return image_paths, image_gt, image_label, image_name


def make_2018_cls_train_dataset(dir='../../../medical_data/ISIC_2018_Skin_Lesion/', mode='training'):
    from data.preprocess.ISICPreprocess_2018 import task3_image_ids, load_task3_training_labels, \
        partition_task3_indices, task3_img_dir
    labels = load_task3_training_labels()
    labels = np.argmax(labels, axis=1)
    train_indices, valid_indices = partition_task3_indices(k=10)
    if mode == "training":
        indices = train_indices
    elif mode == "validation":
        indices = valid_indices
    image_label = labels[indices]
    image_name = [task3_image_ids[i] for i in range(len(task3_image_ids)) if indices[i] == True]
    image_paths = [join(task3_img_dir, task3_image_ids[i] + '.jpg') for i in range(len(task3_image_ids)) if
                   indices[i] == True]
    image_gt = image_paths
    return image_paths, image_gt, image_label, image_name


def make_2018_seg_train_dataset(dir='../../../medical_data/ISIC_2018_Skin_Lesion/', mode='training'):
    from data.preprocess.ISICPreprocess_2018 import task12_image_ids, \
        partition_task1_indices, task1_gt_dir, task12_img_dir
    train_indices, valid_indices = partition_task1_indices(k=10)
    if mode == "training":
        indices = train_indices
    elif mode == "validation":
        indices = valid_indices
    image_name = [task12_image_ids[i] for i in range(len(task12_image_ids)) if indices[i] == True]
    image_paths = [join(task12_img_dir, task12_image_ids[i] + '.jpg') for i in range(len(task12_image_ids)) if
                   indices[i] == True]
    image_gt = [join(task1_gt_dir, task12_image_ids[i] + '_segmentation.png') for i in range(len(task12_image_ids)) if
                indices[i] == True]
    image_label = np.ones(len(image_name))
    return image_paths, image_gt, image_label, image_name


def load_online_isic18seg_data(training_size=2000, dir='../../../medical_data/ISIC_2017_Skin_Lesion/',
                               data_name=ISIC2017):
    """
    Load a data source given it's name
    """
    make_dataset = {
        ISIC2018: make_2018_seg_train_dataset,
    }

    image_paths, image_gt, image_label, image_name = make_dataset[data_name](dir, mode='training')
    final_data = {}
    final_data[DATA_POOL_X] = image_paths
    final_data[DATA_POOL_Y] = image_gt
    final_data[DATA_POOL_Z] = image_label
    final_data[DATA_POOL_N] = image_name
    print(len(image_paths))
    image_paths, image_gt, image_label, image_name = make_dataset[data_name](dir, mode='validation')
    final_data['valid_x'] = image_paths
    final_data['valid_y'] = image_gt
    final_data['valid_z'] = image_label
    final_data['valid_n'] = image_name
    print(len(image_paths))
    final_data, inital_split = split_for_simulation(final_data, training_size=training_size)
    return final_data, inital_split


def load_online_isic_data(training_size=2000, dir='../../../medical_data/ISIC_2017_Skin_Lesion/',
                          data_name=ISIC2017, branch='cls', new_data=False):
    make_dataset = {
        ISIC2017: {
            'cls': make_2017_dataset,
            'seg': make_2017_dataset
        },
        ISIC2018: {
            'cls': make_2018_cls_train_dataset,
            'seg': make_2018_seg_train_dataset
        },
    }
    if branch == 'seg' and new_data:
        image_paths, image_gt, image_label, image_name = make_seg_train_dataset(dir)
    else:
        image_paths, image_gt, image_label, image_name = make_dataset[data_name][branch](dir, mode='training')
    final_data = {}
    final_data[DATA_POOL_X] = image_paths
    final_data[DATA_POOL_Y] = image_gt
    final_data[DATA_POOL_Z] = image_label
    final_data[DATA_POOL_N] = image_name
    print(len(image_paths))
    image_paths, image_gt, image_label, image_name = make_dataset[data_name][branch](dir, mode='validation')
    final_data['valid_x'] = image_paths
    final_data['valid_y'] = image_gt
    final_data['valid_z'] = image_label
    final_data['valid_n'] = image_name
    print(len(image_paths))
    final_data, inital_split = split_for_simulation(final_data, training_size=training_size)
    return final_data, inital_split


def load_isiccls_data(training_size=2000, task_idx=1, output_size=224, data_name=ISIC2017):
    """
    Load a data source given it's name
    """
    if data_name == ISIC2017:
        from data.preprocess.ISICPreprocess_2017 import load_training_data, load_validation_data
        x_train, y_train, z_train = load_training_data(task_idx=task_idx, output_size=output_size)
        x_valid, y_valid, z_valid = load_validation_data(task_idx=task_idx, output_size=output_size)
    if data_name == ISIC2018:
        from data.preprocess.ISICPreprocess_2018 import load_training_data
        (x_train, y_train), (x_valid, y_valid) = load_training_data(task_idx=task_idx, output_size=output_size)
        z_train = y_train
        z_valid = y_valid
    final_data = {}
    print("\t* Loading training data...")
    # load train x
    final_data[DATA_POOL_X] = x_train
    final_data[DATA_POOL_Y] = y_train
    final_data[DATA_POOL_Z] = z_train
    print(x_train.shape)
    print("\t* Loading validation data...")
    # load validation x
    final_data['valid_x'] = x_valid
    final_data['valid_y'] = y_valid
    final_data['valid_z'] = z_valid
    print(x_valid.shape)
    final_data, inital_split = split_for_simulation(final_data, training_size=training_size)
    return final_data, inital_split


def split_for_simulation(all_data, training_size=5000):
    """split initial dataset in train, validation and test sets based on indexes
    """
    index_labelled = np.arange(0, training_size, 1)
    index_unlabelled = np.arange(training_size, len(all_data[DATA_POOL_X]), 1)
    inital_split = {}
    inital_split[INDEX_LABELLED_0] = index_labelled
    inital_split[INDEX_UNLABELLED_0] = index_unlabelled
    inital_split[INDEX_LABELLED] = index_labelled
    inital_split[INDEX_UNLABELLED] = index_unlabelled
    return all_data, inital_split


class ISICDataLoader(data.Dataset):
    """
    Dataloader
    """

    def __init__(self, total_data, split='labelled', labelled_index=None, transform=None, gt_transform=None):
        self.transform = transform
        self.total_data = total_data
        self.gt_trains = gt_transform
        self.spilt = split
        if split not in DATATYPES:
            raise ValueError("not support split type!")
        if split == 'labelled':
            self.images = self.total_data[DATA_POOL_X][labelled_index]
            self.segment = self.total_data[DATA_POOL_Y][labelled_index]
            self.labels = self.total_data[DATA_POOL_Z][labelled_index]
        else:
            self.images = self.total_data[split + '_x']
            self.segment = self.total_data[split + '_y']
            self.labels = self.total_data[split + '_z']
        print("ISICDataLoader " + split + " data size: " + str(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_patch = self.images[index]
        image_segment = self.segment[index]
        image_label = self.labels[index]
        # visualize(np.transpose(image_patch, (1, 2, 0)), join(TMP_DIR, str(index) + "_A_label_train"))
        if self.transform:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image_patch = self.transform(np.transpose(image_patch, (1, 2, 0)))
            if self.gt_trains:
                random.seed(seed)
                torch.manual_seed(seed)
                image_segment = self.gt_trains(np.array(np.squeeze(image_segment), dtype=np.float32))
                # print(torch.unique(image_segment))
                # from driver import mean, std
                # image_patch = np.asarray(image_patch)
                # image_seg = np.asarray(image_segment)
                # visualize(np.transpose(image_patch, (1, 2, 0)) * std + mean, join(TMP_DIR, str(index) + "_img_train"))
                # visualize(np.transpose(image_seg, (1, 2, 0)), join(TMP_DIR, str(index) + "_seg_train"))
        else:
            image_patch = self.transform(np.transpose(image_patch, (1, 2, 0)))
        return {
            "image_patch": image_patch,
            "image_segment": image_segment,
            "image_label": image_label,
        }


class ISICTask1TestDataset(data.Dataset):
    def __init__(self, images, y, z, image_names, image_sizes, transform=None, output_size=224):
        self.transform = transform
        self.labels = z
        self.image_segment = y
        self.images = images
        self.image_names = image_names
        self.image_sizes = image_sizes
        print("ISICTask1TestDataset data size: " + str(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_patch = self.images[index]
        image_label = self.labels[index]
        image_segment = self.image_segment[index]
        image_name = self.image_names[index]
        image_size = self.image_sizes[index]
        # if self.transform:
        #     image_patch = self.transform(np.transpose(image_patch, (1, 2, 0)))
        # else:
        #     image_patch = (image_patch - np.mean(image_patch)) / np.std(image_patch)
        #     image_patch = (image_patch - np.min(image_patch)) / (np.max(image_patch) - np.min(image_patch))
        return {
            "image_patch": image_patch,
            'image_label': image_label,
            'image_segment': image_segment,
            'image_name': image_name,
            'image_size': image_size,
        }


class ISICDataset_plus(data.Dataset):
    def __init__(self, total_data, split='labelled', labelled_index=None, transform=None):
        self.transform = transform
        self.total_data = total_data
        self.spilt = split
        if split not in DATATYPES:
            raise ValueError("not support split type!")
        if split == 'labelled':
            self.images = [self.total_data[DATA_POOL_X][i] for i in range(len(self.total_data[DATA_POOL_X])) if
                           i in labelled_index]
            self.segment = [self.total_data[DATA_POOL_Y][i] for i in range(len(self.total_data[DATA_POOL_Y])) if
                            i in labelled_index]
            self.labels = [self.total_data[DATA_POOL_Z][i] for i in range(len(self.total_data[DATA_POOL_Z])) if
                           i in labelled_index]
            self.name = [self.total_data[DATA_POOL_N][i] for i in range(len(self.total_data[DATA_POOL_N])) if
                         i in labelled_index]
        else:
            self.images = self.total_data[split + '_x']
            self.segment = self.total_data[split + '_y']
            self.labels = self.total_data[split + '_z']
            self.name = self.total_data[split + '_n']
        print("ISICDataLoader " + split + " data size: " + str(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_patch = self.images[index]
        image_segment = self.segment[index]
        image_label = self.labels[index]
        image_name = self.name[index]
        image = self.pil_loader(image_patch, isGT=False)
        gt = self.pil_loader(image_segment, isGT=True)
        if self.transform:
            image, gt = self.transform(image, gt)
        return {
            "image_patch": image,
            'image_label': image_label,
            'image_segment': gt,
            'image_name': image_name,
        }

    def pil_loader(self, path, isGT=False):
        with open(path, 'rb') as f:
            img = Image.open(f)
            mode = 'RGB' if not isGT else 'L'
            return img.convert(mode)


class ISIC2018ClsTestAugDataset(data.Dataset):
    def __init__(self, path='../../../medical_data/ISIC_2018_Skin_Lesion/', transform=None,
                 ):
        self.path = path
        from data.preprocess.ISICPreprocess_2018 import task3_test_image_ids, task3_test_img_dir
        image_paths = [join(task3_test_img_dir, task3_test_image_ids[i] + '.jpg') for i in
                       range(len(task3_test_image_ids))]

        self.samples = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img_name = img_path.split("/")[-1]
        image = self.pil_loader(img_path)
        if self.transform:
            image = self.transform(image)

        return image, img_name

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

class ISIC2017ClsTestAugDataset(data.Dataset):
    def __init__(self, path='../../../medical_data/ISIC_2017_Skin_Lesion/', transform=None):
        self.path = path
        self.samples = self.make_2017_dataset(path)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, melanoma, seborrheic_keratosis = self.samples[idx]
        img_name = img_path.split("/")[-1]
        image = self.pil_loader(img_path)
        if self.transform:
            image = self.transform(image)
        # labels = np.array(
        #     [int(melanoma), int(seborrheic_keratosis), 1 - int(melanoma) - int(seborrheic_keratosis)])
        # return image, torch.from_numpy(np.array(np.argmax(labels))), img_name

        return image, torch.from_numpy(np.array(
            np.argmax([melanoma, seborrheic_keratosis, 1 - melanoma - seborrheic_keratosis]))), img_name

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def make_2017_dataset(self, dir):
        images = []
        img_dir = os.path.join(dir, "ISIC-2017_Test_v2_Data")
        csv_filename = os.path.join(dir, "ISIC-2017_Test_v2_Part3_GroundTruth.csv")
        label_list = pd.read_csv(csv_filename)
        for index, row in label_list.iterrows():
            images.append(
                (os.path.join(img_dir, row["image_id"] + ".jpg"), row["melanoma"], row["seborrheic_keratosis"]))
        print("data_length:" + str(len(images)))
        return images


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data.crop_transform import argumentation_train
    from torchvision.utils import make_grid, save_image
    from driver import std, mean


    def read_data(dataloader):
        while True:
            for batch in dataloader:
                image = batch['image_patch']
                label = batch['image_label']
                segment = batch['image_segment']
                yield {
                    'image_patch': image,
                    'image_label': label,
                    'image_segment': segment,
                }


    def merge_batch(batch):
        image_patch = [torch.unsqueeze(image, dim=0) for inst in batch for image in inst["image_patch"]]
        image_label = [inst["image_label"] for inst in batch for _ in inst["image_patch"]]
        image_segment = [torch.unsqueeze(image, dim=0) for inst in batch for image in inst["image_segment"]]
        image_patch = torch.cat(image_patch, dim=0)
        image_label = torch.tensor(image_label)
        image_segment = torch.cat(image_segment, dim=0)
        image_name = [inst["image_name"] for inst in batch for _ in inst["image_patch"]]

        return {"image_patch": image_patch,
                "image_label": image_label,
                "image_segment": image_segment,
                "image_name": image_name
                }


    data_sets, inital_split = load_online_isic_data(training_size=10015,
                                                    dir='../../medical_data/ISIC_2018_Skin_Lesion/',
                                                    data_name=ISIC2018)
    test_dataset = ISICDataset_plus(data_sets, split='labelled', labelled_index=inital_split['index_labelled'],
                                    transform=argumentation_train(output_size=224))
    test_loader_mtl_arl = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=merge_batch)
    data_iter = read_data(test_loader_mtl_arl)
    # for i in range(10):
    #     batch_gen = next(data_iter)
    #     image_mtl = batch_gen['image_patch']
    #     label_mtl = batch_gen['image_label']
    #     seg_mtl = batch_gen['image_segment']
    #     grid = make_grid(image_mtl, nrow=4, padding=2)
    #     visualize(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)) * std + mean,
    #               join('./tmp', str(i) + "_images"))
    #     grid = make_grid(seg_mtl, nrow=4, padding=2)
    #     visualize(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)),
    #               join('./tmp', str(i) + "_gt"))
