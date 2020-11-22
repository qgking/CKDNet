# -*- coding: utf-8 -*-
# @Time    : 19/11/8 9:54
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : base_train_helper.py
from commons.utils import *
from tensorboardX import SummaryWriter
import torch
from models import MODELS
import matplotlib.pyplot as plt
from data.ISICDataLoader import ISICDataLoader, load_isiccls_data, load_online_isic_data
from data.preprocess.ISICPreprocess_2017 import mkdir_if_not_exist
from driver import active_transform
from torch.utils.data import DataLoader
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import label as sk_label
from data.crop_transform import argumentation_val, argumentation_val_normal, argumentation_train, argumentation_test, \
    argumentation_train_seg_aug, argumentation_train_normal
import torchvision.transforms as tvt
from skimage import io
from data.preprocess.ISICPreprocess_2017 import task1_test_gt_dir
import pickle
from data import DATA_POOL_X, DATA_POOL_Y, INDEX_LABELLED, ISIC2017, ISIC2018
from commons.evaluation import compute_all_metric_for_single_seg, \
    compute_all_metric_for_class_wise_cls
from data.ISICDataLoader import ISIC2017ClsTestAugDataset, ISICDataset_plus, \
    ISIC2018ClsTestAugDataset

plt.rcParams.update({'figure.max_open_warning': 20})


class BaseTrainHelper(object):
    def __init__(self, model, criterions, config):
        self.model = model
        self.criterions = criterions
        self.config = config
        # p = next(filter(lambda p: p.requires_grad, generator.parameters()))
        self.use_cuda = config.use_cuda
        # self.device = p.get_device() if self.use_cuda else None
        self.device = config.gpu if self.use_cuda else None
        if self.config.train:
            self.make_dirs()
        self.define_log()
        self.out_put_summary()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

    def save_checkpoint(self, state, filename='checkpoint.pth'):
        torch.save(state, filename)

    def merge_batch(self, batch):
        image_patch = [torch.unsqueeze(image, dim=0) for inst in batch for image in inst["image_patch"]]
        orders = np.random.permutation(len(image_patch))
        image_patch = [image_patch[o] for o in orders]
        image_patch = torch.cat(image_patch, dim=0)
        image_label = [inst["image_label"] for inst in batch for _ in inst["image_patch"]]
        image_label = [image_label[o] for o in orders]
        image_label = torch.tensor(image_label)
        image_segment = [torch.unsqueeze(image, dim=0) for inst in batch for image in inst["image_segment"]]
        image_segment = [image_segment[o] for o in orders]
        image_segment = torch.cat(image_segment, dim=0)
        image_name = [inst["image_name"] for inst in batch for _ in inst["image_patch"]]
        image_name = [image_name[o] for o in orders]

        return {"image_patch": image_patch,
                "image_label": image_label,
                "image_segment": image_segment,
                "image_name": image_name
                }

    def make_dirs(self):
        if not isdir(self.config.tmp_dir):
            os.makedirs(self.config.tmp_dir)
        if not isdir(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        if not isdir(self.config.save_model_path):
            os.makedirs(self.config.save_model_path)
        if not isdir(self.config.tensorboard_dir):
            os.makedirs(self.config.tensorboard_dir)
        if not isdir(self.config.submission_dir):
            os.makedirs(self.config.submission_dir)

    def reset_model(self):
        del self.model
        self.model = MODELS[self.config.model](backbone=self.config.backbone, cls_branch=self.config.cls_branch,
                                               seg_branch=self.config.seg_branch,
                                               cls_num_classes=self.config.cls_classes, seg_num_class=1)
        if self.use_cuda and self.model:
            self.model.to(self.equipment)
            if len(self.config.gpu_count) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_count)

    def read_data(self, dataloader):
        while True:
            for batch in dataloader:
                image = batch['image_patch']
                label = batch['image_label']
                segment = batch['image_segment']
                image = image.to(self.equipment).float()
                label = label.to(self.equipment).long()
                segment = segment.to(self.equipment).float()
                yield {
                    'image_patch': image,
                    'image_label': label,
                    'image_segment': segment,
                }

    def define_log(self):
        if self.config.train:
            self.log = Logger(self.config.log_file)
        else:
            self.log = Logger(join(self.config.save_dir, 'test_log.txt'))
        sys.stdout = self.log

    def move_to_cuda(self):
        if self.use_cuda and self.model:
            torch.cuda.set_device(self.config.gpu)
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.equipment)
            for key in self.criterions.keys():
                print(key)
                self.criterions[key].to(self.equipment)
            if len(self.config.gpu_count) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_count)
        else:
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_model_checkpoint(self, epoch, optimizer):
        save_file = join(self.config.save_model_path, 'checkpoint_epoch_%03d.pth' % (epoch + 1))
        self.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=save_file)

    def save_model_iter_checkpoint(self, iter, optimizer):
        save_file = join(self.config.save_model_path, 'checkpoint_iter_%03d.pth' % (iter))
        self.save_checkpoint({
            'iter': iter,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=save_file)

    def save_best_checkpoint(self, model_optimizer=None, save_model=False, iter=0):
        opti_file_path = join(self.config.save_model_path, "iter_" + str(iter) + "_best_optim.opt")
        save_model_path = join(self.config.save_model_path, "iter_" + str(iter) + "_best_model.pt")
        if save_model:
            torch.save(self.model.state_dict(), save_model_path)
        if model_optimizer is not None:
            torch.save(model_optimizer.state_dict(), opti_file_path)

    def get_best_checkpoint(self, model_optimizer=None, load_model=False, iter=0):
        if model_optimizer is not None:
            load_file = join(self.config.save_model_path, "iter_" + str(iter) + "_best_optim.opt")
        if load_model:
            load_file = join(self.config.save_model_path, "iter_" + str(iter) + "_best_model.pt")
        print('load file ' + load_file)
        state_dict = torch.load(load_file, map_location={'cuda:0': 'cuda:' + str(self.config.gpu),
                                                         'cuda:1': 'cuda:' + str(self.config.gpu),
                                                         'cuda:2': 'cuda:' + str(self.config.gpu),
                                                         'cuda:3': 'cuda:' + str(self.config.gpu)})
        # from collections import OrderedDict
        # if 'module' not in list(state_dict.keys())[0][:7]:
        #     return state_dict
        # else:
        #     new_state_dict = OrderedDict()
        #     for k, v in state_dict.items():
        #         name = k[7:]  # remove `module.`
        #         new_state_dict[name] = v
        #     return new_state_dict
        return state_dict

    def load_best_optim(self, optim, iter=0):
        state_dict_file = self.get_best_checkpoint(model_optimizer=True, iter=iter)
        optim.load_state_dict(state_dict_file)
        return optim

    def load_best_state(self, iter=0):
        state_dict_file = self.get_best_checkpoint(load_model=True, iter=iter)
        # for k, v in state_dict_file.items():
        #     print(k)
        # for k, v in self.model.state_dict().items():
        #     print(k)
        self.model.load_state_dict(state_dict_file)

    def out_put_summary(self):
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)

    def write_summary(self, epoch, criterions):
        for key in criterions.keys():
            self.summary_writer.add_scalar(
                key, criterions[key], epoch)

    def plot_vali_loss(self, iter, epoch, criterions, type='vali'):
        if epoch == 0:
            self.seg_loss_cal = []
            self.cls_loss_cal = []
            self.seg_acc_cal = []
            self.cls_acc_cal = []
        plt.figure(figsize=(16, 10), dpi=100)
        self.cls_loss_cal.append(criterions[type + '/cls_loss'])
        self.seg_loss_cal.append(criterions[type + '/seg_loss'])
        self.cls_acc_cal.append(criterions[type + '/cls_acc'])
        self.seg_acc_cal.append(criterions[type + '/seg_acc'])
        epochs = range(len(self.seg_loss_cal))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.cls_loss_cal, color='red', marker='o', linestyle='-', label='cls loss')  # 'bo'为画蓝色圆点，不连线
        plt.plot(epochs, self.seg_loss_cal, color='blue', marker='s', linestyle='-', label='seg loss')
        max_ylim = 3 if type == 'vali' else 3
        plt.ylim(0, max_ylim)
        plt.xlim(-2, self.config.epochs + 5)
        plt.title(type + ' loss vs. epoches')
        plt.ylabel('loss')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.cls_acc_cal, color='darkgreen', marker='.', linestyle='-', label='cls accuracy')
        plt.plot(epochs, self.seg_acc_cal, color='orange', marker='D', linestyle='-', label='seg accuracy')
        plt.ylim(0.5, 1)
        plt.xlim(-2, self.config.epochs + 5)
        plt.xlabel(type + ' accuracy vs. epoches')
        plt.ylabel(type + ' accuracy')
        plt.legend()
        plt.savefig(join(self.config.submission_dir, "iter_" + str(iter) + '_' + type + "_accuracy_loss.jpg"))
        plt.close()

    def plot_train_loss(self, iter, epoch, criterions, type='train'):
        if epoch == 0:
            self.train_seg_loss_cal = []
            self.train_cls_loss_cal = []
            self.train_seg_acc_cal = []
            self.train_cls_acc_cal = []
        plt.figure(figsize=(16, 10), dpi=100)
        self.train_cls_loss_cal.append(criterions[type + '/cls_loss'])
        self.train_seg_loss_cal.append(criterions[type + '/seg_loss'])
        self.train_cls_acc_cal.append(criterions[type + '/cls_acc'])
        self.train_seg_acc_cal.append(criterions[type + '/seg_acc'])
        epochs = range(len(self.train_seg_loss_cal))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.train_cls_loss_cal, color='red', marker='o', linestyle='-',
                 label='cls loss')  # 'bo'为画蓝色圆点，不连线
        plt.plot(epochs, self.train_seg_loss_cal, color='blue', marker='s', linestyle='-', label='seg loss')
        plt.ylim(0, 2)
        plt.xlim(-2, self.config.epochs + 5)
        plt.title(type + ' loss vs. epoches')
        plt.ylabel('loss')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.train_cls_acc_cal, color='darkgreen', marker='.', linestyle='-', label='cls accuracy')
        plt.plot(epochs, self.train_seg_acc_cal, color='orange', marker='D', linestyle='-', label='seg accuracy')
        plt.ylim(0.5, 1)
        plt.xlim(-2, self.config.epochs + 5)
        plt.xlabel(type + ' accuracy vs. epoches')
        plt.ylabel(type + ' accuracy')
        plt.legend()
        plt.savefig(join(self.config.submission_dir, "iter_" + str(iter) + '_' + type + "_accuracy_loss.jpg"))
        plt.close()


    # for isic normal cls/seg
    def get_data_loader(self, global_transform=None, gt_transform=None, default_label_size=2000,
                        train_batch_size=8,
                        test_batch_size=8, output_size=224, task=3):
        data_sets, inital_split = load_isiccls_data(default_label_size,
                                                    output_size=output_size, data_name=self.config.data_name,
                                                    task_idx=task)
        train_dataset = ISICDataLoader(data_sets, split='labelled',
                                       labelled_index=inital_split[INDEX_LABELLED], transform=global_transform,
                                       gt_transform=gt_transform)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

        vali_dataset = ISICDataLoader(data_sets, split='valid', transform=active_transform)
        vali_loader = DataLoader(vali_dataset, batch_size=test_batch_size, shuffle=False)
        data_sets = {
            DATA_POOL_X: data_sets[DATA_POOL_X],
            DATA_POOL_Y: data_sets[DATA_POOL_Y]
        }
        return train_loader, vali_loader, data_sets, inital_split

    # for isic 2017 aug cls/seg and for isic 2018 aug cls
    def get_aug_data_loader(self, branch='cls', online_aug='online', vali_aug='normal', default_label_size=2000,
                            train_batch_size=8,
                            test_batch_size=8, new_data=False):
        train_augmentation = {
            'cls': {
                'online': argumentation_train,
                'normal': argumentation_train_normal
            },
            'seg': {
                'online': argumentation_train_seg_aug,
                'normal': argumentation_train_normal,
            }
        }
        vali_augmentation = {
            'online': argumentation_val,
            'normal': argumentation_val_normal
        }
        data_sets, inital_split = load_online_isic_data(default_label_size, data_name=self.config.data_name,
                                                        branch=branch, new_data=new_data)
        # data_sets, inital_split = load_mix_online_isic_data(default_label_size, data_name=self.config.data_name,
        #                                                     branch=branch)
        train_dataset = ISICDataset_plus(data_sets, split='labelled',
                                         labelled_index=inital_split[INDEX_LABELLED],
                                         transform=train_augmentation[branch][online_aug](
                                             output_size=self.config.patch_x))
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                  collate_fn=self.merge_batch,
                                  shuffle=True, num_workers=self.config.workers)
        vali_dataset = ISICDataset_plus(data_sets, split='valid',
                                        transform=vali_augmentation[vali_aug](
                                            output_size=self.config.patch_x))
        vali_loader = DataLoader(vali_dataset, batch_size=test_batch_size,
                                 collate_fn=self.merge_batch, shuffle=False,
                                 num_workers=self.config.workers)
        # vali_dataset = ISICDataLoader(data_sets, split='valid', transform=active_transform)
        # vali_loader = DataLoader(vali_dataset, batch_size=self.config.test_cls_batch_size, shuffle=False)
        data_sets = {
            DATA_POOL_X: data_sets[DATA_POOL_X],
            DATA_POOL_Y: data_sets[DATA_POOL_Y]
        }

        return train_loader, vali_loader, data_sets, inital_split

    def get_cls_test_aug_data_loader(self):
        DATA_SET = {
            ISIC2017: ISIC2017ClsTestAugDataset,
            ISIC2018: ISIC2018ClsTestAugDataset,
        }
        test_dataset = DATA_SET[self.config.data_name](
            transform=argumentation_test(output_size=self.config.patch_x))
        test_loader_mtl = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=self.config.workers)
        return test_loader_mtl


    def get_seg_test_data_loader(self):
        from data.ISICDataLoader import ISICTask1TestDataset
        if self.config.data_name == ISIC2017:
            from data.preprocess.ISICPreprocess_2017 import load_test_data
            images, y, z, image_names, image_sizes = load_test_data(task_idx=1, output_size=self.config.patch_x)
        if self.config.data_name == ISIC2018:
            from data.preprocess.ISICPreprocess_2018 import load_test_data
            images, image_names, image_sizes = load_test_data(task_idx=1, output_size=self.config.patch_x)
            y = images
            z = images
        test_dataset = ISICTask1TestDataset(images, y, z, image_names, image_sizes, transform=active_transform,
                                            output_size=self.config.patch_x)
        test_loader_seg = DataLoader(test_dataset, batch_size=self.config.test_seg_batch_size, shuffle=False)
        return test_loader_seg


    def evaluation_cls(self, y, y_score, ac_iter, type='normal', thres=0.5):
        y_score = np.array(y_score)
        lesion_cls_metrics = compute_all_metric_for_class_wise_cls(y, y_score, thres=thres)
        pickle.dump(lesion_cls_metrics,
                    open(join(self.config.submission_dir, "cls_eval_" + str(ac_iter) + "_" + type), 'wb'))
        pickle.dump([y, y_score],
                    open(join(self.config.submission_dir, "cls_eval_detail_" + str(ac_iter) + "_" + type), 'wb'))

    def evaluation_seg(self, pred_label, image_names, image_sizes, ac_iter, threshold=0.5):
        y_pred = pred_label
        self.create_seg_submit_files(image_names=image_names, y_pred=y_pred, image_sizes=image_sizes,
                                     ac_iter=ac_iter, threshold=threshold)
        # y_pred = self.task1_post_process(y_prediction=y_pred, threshold=threshold, gauss_sigma=2.)
        # self.create_seg_submit_files(image_names=image_names, y_pred=y_pred, image_sizes=image_sizes,
        #                              ac_iter=ac_iter, threshold=threshold)

    def task1_post_process(self, y_prediction, threshold=0.5, gauss_sigma=0.):
        for im_index in range(len(y_prediction)):
            # smooth image by Gaussian filtering
            if gauss_sigma > 0.:
                y_prediction[im_index] = gaussian_filter(input=y_prediction[im_index], sigma=gauss_sigma)

            thresholded_image = y_prediction[im_index] > threshold
            # find largest connected component
            labels, num_labels = sk_label(thresholded_image, return_num=True)
            max_label_idx = -1
            max_size = 0
            for label_idx in range(0, num_labels + 1):

                if np.sum(thresholded_image[labels == label_idx]) == 0:
                    continue

                current_size = np.sum(labels == label_idx)

                if current_size > max_size:
                    max_size = current_size
                    max_label_idx = label_idx

            if max_label_idx > -1:
                y_prediction[im_index] = labels == max_label_idx
            else:  # no predicted pixels found
                y_prediction[im_index] = y_prediction[im_index] * 0

        return y_prediction

    def evaluation_seg_2018(self, pred_label, image_names, image_sizes, ac_iter, threshold=0.5):
        y_pred = pred_label
        # self.create_seg_sumbit_files_isic2018(image_names=image_names, y_pred=y_pred, image_sizes=image_sizes,
        #                                       ac_iter=ac_iter, threshold=threshold)
        y_pred = self.task1_post_process(y_prediction=y_pred, threshold=threshold, gauss_sigma=2.)
        self.create_seg_sumbit_files_isic2018(image_names=image_names, y_pred=y_pred, image_sizes=image_sizes,
                                              ac_iter=ac_iter, threshold=threshold, tag='postprocess')

    def create_seg_sumbit_files_isic2018(self, image_names, y_pred, image_sizes, ac_iter, threshold=0.5, tag='normal'):
        output_dir = self.config.submission_dir + '/' + self.config.model + '_ac_iter_' + str(
            ac_iter) + '_task1_test_' + tag + '_thres_' + str(
            threshold)
        mkdir_if_not_exist([output_dir])
        for i_image, i_name in enumerate(image_names):
            # print(i_name)
            current_pred = np.squeeze(y_pred[i_image], axis=0)
            ttest = tvt.Compose([
                tvt.ToPILImage(),
                tvt.Resize(image_sizes[i_image]),
            ])
            resized_pred_tensor = ttest(torch.from_numpy(current_pred))
            resized_pred_tensor = np.array(resized_pred_tensor) / 255.
            resized_pred_tensor[resized_pred_tensor > threshold] = 255
            resized_pred_tensor[resized_pred_tensor <= threshold] = 0
            im = Image.fromarray(resized_pred_tensor.astype(np.uint8))
            im.save(output_dir + '/' + i_name + '_segmentation.png')
        print("----------- Done save -----------")
        zipDir(output_dir, output_dir + '.zip')
        print("----------- Done zip -----------")

    def create_seg_submit_files(self, image_names, y_pred, image_sizes, ac_iter, threshold=0.5):
        output_dir = self.config.submission_dir + '/' + self.config.model + '_ac_iter_' + str(
            ac_iter) + '_task1_test_thres_' + str(threshold)
        mkdir_if_not_exist([output_dir])
        segmentation_metrics = {
            # 'ROC': 0,
            'Jaccard': 0,
            'F1': 0, 'ACCURACY': 0, 'SENSITIVITY': 0, 'SPECIFICITY': 0,
            # 'PRECISION': 0,
            'DICESCORE': 0}
        lesion_segmentation_scores = {}
        for i_image, i_name in enumerate(image_names):
            # if i_image>1:
            #     break
            # print(i_name)
            current_pred = np.squeeze(y_pred[i_image], axis=0)
            # current_pred = current_pred * 255
            image_gt = np.asarray(io.imread(os.path.join(task1_test_gt_dir, i_name + '_segmentation.png')) // 255)
            ttest = tvt.Compose([
                tvt.ToPILImage(),
                tvt.Resize(image_sizes[i_image]),
            ])
            resized_pred_tensor = ttest(torch.from_numpy(current_pred))
            resized_pred_tensor = np.array(resized_pred_tensor) / 255.
            resized_pred_tensor[resized_pred_tensor > threshold] = 1
            resized_pred_tensor[resized_pred_tensor <= threshold] = 0
            # MAP = do_crf(img, pred, zero_unsure=False)
            # plt.imshow(MAP)
            # im = Image.fromarray((resized_pred_tensor * 255.).astype(np.uint8))
            # im.save(output_dir + '/' + i_name + '_segmentation.png')
            scores = compute_all_metric_for_single_seg(image_gt, resized_pred_tensor)
            # if not isdir(join(self.config.tmp_dir, 'couter/')):
            #     makedirs(join(self.config.tmp_dir, 'couter/'))
            # draw_contour_on_image(join(task1_test_img_dir, i_name + '.jpg'), image_gt, resized_pred_tensor,
            #                       join(self.config.tmp_dir, 'couter/'))

            for metric in segmentation_metrics:
                if metric not in lesion_segmentation_scores:
                    lesion_segmentation_scores[metric] = []
                lesion_segmentation_scores[metric].extend(scores[metric])
        lesion_segmentation_metrics = {}
        info = ''
        for m in lesion_segmentation_scores:
            lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
            info += ('\n' + m + ': {val:.9f}   '.format(val=lesion_segmentation_metrics[m]))
        zipDir(output_dir, output_dir + '.zip')
        pickle.dump(lesion_segmentation_metrics,
                    open(join(self.config.submission_dir, "seg_eval_" + str(ac_iter)), 'wb'))
        pickle.dump(lesion_segmentation_scores,
                    open(join(self.config.submission_dir, "seg_eval_detail_" + str(ac_iter)), 'wb'))
        print(info)

    def create_isic_2018_cls_submit_csv(self, image_names, y_prob, ac_iter, tag='normal'):
        print('---- Done predicting -- creating submission---- ')
        submission_file = self.config.submission_dir + '/ac_iter_' + str(
            ac_iter) + '_task3_test_submission_' + tag + '.csv'
        f = open(submission_file, 'w')
        f.write('image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n')
        for i_image, i_name in enumerate(image_names):
            i_line = i_name
            for i_cls in range(7):
                prob = y_prob[i_image, i_cls]
                if prob < 0.001:
                    prob = 0.
                i_line += ',' + str(prob)
            i_line += '\n'
            f.write(i_line)  # Give your csv text here.
        f.close()
        print("----------- Done save -----------")

    def get_model_backbone_feature(self, batch):
        return self.model.backbone(batch)

    def generate_batch(self, batch):
        images = batch['image_patch'].to(self.equipment).float()
        segment = batch['image_segment'].to(self.equipment).float()
        labels = batch['image_label'].to(self.equipment).long()
        return images, segment, labels
