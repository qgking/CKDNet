import matplotlib

matplotlib.use("Agg")
import sys

sys.path.extend(["../../", "../", "./"])
from commons.utils import *
import torch
import random
from driver.gen_driver.ConfigGEN import Configurable
from driver.gen_driver.train_waterfall_helper import MUTUALHelper
import configparser
from models.seg_models.deeplab import DeepLab_Aux
from models import MODELS
import pandas as pd
from driver import std, mean
import torchvision.transforms.functional as transform_func
import torchvision.transforms as tvt
from data import ISIC2018, ISIC2017
import time

config = configparser.ConfigParser()
import argparse


def main(config):
    model_seg_coarse = DeepLab_Aux(num_classes=1, return_features=True)
    model_cls = MODELS['Integrate_Model_Cls_Ensemble_CAM_Att'](backbone=config.backbone, num_classes=config.cls_classes)
    model_seg = MODELS[config.model](backbone=config.backbone, n_channels=3, num_classes=1)
    criterion = {
    }
    mutual_helper = MUTUALHelper(model_seg, criterion,
                                 config)
    mutual_helper.move_to_cuda()
    print("data name ", mutual_helper.config.data_name)
    print("data size ", mutual_helper.config.patch_x)
    test_loader_seg = mutual_helper.get_seg_test_data_loader()
    predict_fun = {
        ISIC2017: predict_isic17,
        ISIC2018: predict_isic18
    }
    try:
        model_seg_coarse = mutual_helper.load_pretrained_coarse_seg_model(model_seg_coarse)
        model_seg_coarse.eval()
        model_seg_coarse.to(mutual_helper.equipment)
    except FileExistsError as e:
        print(e)
        exit(0)

    try:
        model_cls = mutual_helper.load_pretrained_cls_model(model_cls)
        model_cls.eval()
        model_cls.to(mutual_helper.equipment)
    except FileExistsError as e:
        print(e)
        exit(0)

    for nb_acl_iter in range(mutual_helper.config.nb_active_learning_iter):
        print("\n-----------load best state of model -----------")
        mutual_helper.load_best_state(iter=nb_acl_iter)
        predict_fun[mutual_helper.config.data_name](mutual_helper, model_seg_coarse, model_cls, test_loader_seg,
                                                    nb_acl_iter)
        mutual_helper.log.flush()
        exit(0)

    print("\n----------- TRAINING FINISHED -----------")
    mutual_helper.summary_writer.close()


def predict_isic17(mutual_helper, model_seg_coarse, model_cls, test_loader_seg, ac_iter):
    mutual_helper.model.eval()
    pred_label = []
    image_names = []
    image_sizes = []
    gt_labels = []
    pre_transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize(mean=mean,
                      std=std)
    ])
    rotation_angles = [90, 180, 270, 0]
    with torch.no_grad():
        for i, batch in enumerate(test_loader_seg):
            images, labels, _ = mutual_helper.generate_batch(batch)
            image_name = batch['image_name']
            image_size = batch['image_size'].detach().cpu().numpy()
            for ii in range(len(images)):
                images_one_trans = []
                img_tmp = images[ii].detach().cpu().numpy().astype(np.uint8)
                if ii <= 10:
                    start = time.time()
                img_pil = Image.fromarray(np.transpose(img_tmp, (1, 2, 0)))
                # img_pil.save(join(mutual_helper.config.tmp_dir, str(i) + "_img_0.png"))
                img_pil_trans = transform_func.hflip(img_pil)
                img_tensor = pre_transform(img_pil_trans).unsqueeze(0).to(mutual_helper.equipment)
                prob = infer_img(mutual_helper, model_seg_coarse, model_cls, img_tensor)
                res = prob.detach().cpu().squeeze().numpy() * 255.
                prob_pil = Image.fromarray(res.astype(np.uint8))
                prob_pil_trans = transform_func.hflip(prob_pil)
                # prob_pil_trans.save(join(mutual_helper.config.tmp_dir, str(i) + "_probs_1.png"))
                images_one_trans.append(np.array(prob_pil_trans))

                img_tmp = images[ii].detach().cpu().numpy().astype(np.uint8)
                img_pil = Image.fromarray(np.transpose(img_tmp, (1, 2, 0)))
                img_pil_trans = transform_func.vflip(img_pil)
                img_tensor = pre_transform(img_pil_trans).unsqueeze(0).to(mutual_helper.equipment)
                prob = infer_img(mutual_helper, model_seg_coarse, model_cls, img_tensor)
                res = prob.detach().cpu().squeeze().numpy() * 255.
                prob_pil = Image.fromarray(res.astype(np.uint8))
                prob_pil_trans = transform_func.vflip(prob_pil)
                # prob_pil_trans.save(join(mutual_helper.config.tmp_dir, str(i) + "_probs_11.png"))
                images_one_trans.append(np.array(prob_pil_trans))

                for vv in range(len(rotation_angles)):
                    angle = rotation_angles[vv]
                    img_tmp = images[ii].detach().cpu().numpy().astype(np.uint8)
                    img_pil = Image.fromarray(np.transpose(img_tmp, (1, 2, 0)))
                    img_pil_trans = transform_func.rotate(img_pil, angle)
                    # img_pil_trans.save(join(mutual_helper.config.tmp_dir, str(i) + "_img_12.png"))
                    img_tensor = pre_transform(img_pil_trans).unsqueeze(0).to(mutual_helper.equipment)
                    prob = infer_img(mutual_helper, model_seg_coarse, model_cls, img_tensor)
                    res = prob.detach().cpu().squeeze().numpy() * 255.
                    prob_pil = Image.fromarray(res.astype(np.uint8))
                    # prob_pil.save(join(mutual_helper.config.tmp_dir, str(i) + "_probs_02.png"))
                    prob_pil_trans = transform_func.rotate(prob_pil, 0 - angle)
                    # prob_pil_trans.save(join(mutual_helper.config.tmp_dir, str(i) + "_probs_12.png"))
                    images_one_trans.append(np.array(prob_pil_trans))
                if ii <= 10:
                    print('Time collapse for each img: %s' % (time.time() - start))

                images_one_trans = np.stack(images_one_trans)
                probs = np.sum(images_one_trans, axis=0) / len(images_one_trans)
                probs = probs / 255.
                pred_label.append(probs)
                gt_labels.append(labels[ii].detach().cpu().numpy())
                image_names.append(image_name[ii])
                image_sizes.append(image_size[ii])

                # img_tmp = images[ii].detach().cpu().numpy().astype(np.uint8)
                # img_pil = Image.fromarray(np.transpose(img_tmp, (1, 2, 0)))
                # img_pil.save(join(mutual_helper.config.tmp_dir, image_name[ii] + ".png"))
                # if not isdir(join(mutual_helper.config.tmp_dir, 'probs')):
                #     makedirs(join(mutual_helper.config.tmp_dir, 'probs'))
                # visualize(np.expand_dims(probs, axis=-1),
                #           join(join(mutual_helper.config.tmp_dir, 'probs'), image_name[ii] + "_probs"))
                #
                # if not isdir(join(mutual_helper.config.tmp_dir, 'entropy')):
                #     makedirs(join(mutual_helper.config.tmp_dir, 'entropy'))
                # coarse = np.stack([1 - probs, probs])
                # num_classes = coarse.shape[0]
                # entropy_map = np.zeros((coarse.shape[1], coarse.shape[2]))
                # for c in range(num_classes):
                #     entropy_map = entropy_map - (coarse[c, :, :] * np.log2(coarse[c, :, :] + 1e-12))
                # entropy_map = np.expand_dims(entropy_map, axis=-1)
                # visualize(entropy_map, join(join(mutual_helper.config.tmp_dir, 'entropy'), image_name[ii] + "entropy"))
    pred_label = np.array(pred_label)
    pred_label = np.expand_dims(pred_label, axis=1).astype(np.float32)
    mutual_helper.evaluation_seg(pred_label, image_names, image_sizes, ac_iter, threshold=0.5)


def infer_img(mutual_helper, model_seg_coarse, model_cls, images):
    images_cls_logits, seg_backbone_out, _, _ = model_seg_coarse(images)
    probs_cls = torch.sigmoid(images_cls_logits)
    cls_features_out = model_cls.get_backbone_out(images, probs_cls, seg_backbone_out)
    # _, _, cam = mutual_helper.generate_cam_ex_batch(model_cls, images.detach(), probs_cls, seg_backbone_out)
    cam = None
    predictions_seg = mutual_helper.model(images, cam, cls_features_out, dua=False)
    probs = torch.sigmoid(predictions_seg)
    return probs


def predict_isic18(mutual_helper, model_seg_coarse, model_cls, test_loader_seg, ac_iter):
    mutual_helper.model.eval()
    pred_label = []
    image_names = []
    image_sizes = []
    pre_transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize(mean=mean,
                      std=std)
    ])
    rotation_angles = [90, 180, 270, 0]
    with torch.no_grad():
        for i, batch in enumerate(test_loader_seg):
            images = batch['image_patch'].to(mutual_helper.equipment).float()
            image_name = batch['image_name']
            image_size = batch['image_size'].detach().cpu().numpy()
            for ii in range(len(images)):
                images_one_trans = []
                img_tmp = images[ii].detach().cpu().numpy().astype(np.uint8)
                img_pil = Image.fromarray(np.transpose(img_tmp, (1, 2, 0)))
                # img_pil.save(join(mutual_helper.config.tmp_dir, str(i) + "_img_0.png"))
                img_pil_trans = transform_func.hflip(img_pil)
                img_tensor = pre_transform(img_pil_trans).unsqueeze(0).to(mutual_helper.equipment)
                prob = infer_img(mutual_helper, model_seg_coarse, model_cls, img_tensor)
                res = prob.detach().cpu().squeeze().numpy() * 255.
                prob_pil = Image.fromarray(res.astype(np.uint8))
                prob_pil_trans = transform_func.hflip(prob_pil)
                # prob_pil_trans.save(join(mutual_helper.config.tmp_dir, str(i) + "_probs_1.png"))
                images_one_trans.append(np.array(prob_pil_trans))

                img_tmp = images[ii].detach().cpu().numpy().astype(np.uint8)
                img_pil = Image.fromarray(np.transpose(img_tmp, (1, 2, 0)))
                img_pil_trans = transform_func.vflip(img_pil)
                img_tensor = pre_transform(img_pil_trans).unsqueeze(0).to(mutual_helper.equipment)
                prob = infer_img(mutual_helper, model_seg_coarse, model_cls, img_tensor)
                res = prob.detach().cpu().squeeze().numpy() * 255.
                prob_pil = Image.fromarray(res.astype(np.uint8))
                prob_pil_trans = transform_func.vflip(prob_pil)
                # prob_pil_trans.save(join(mutual_helper.config.tmp_dir, str(i) + "_probs_11.png"))
                images_one_trans.append(np.array(prob_pil_trans))

                for vv in range(len(rotation_angles)):
                    angle = rotation_angles[vv]
                    img_tmp = images[ii].detach().cpu().numpy().astype(np.uint8)
                    img_pil = Image.fromarray(np.transpose(img_tmp, (1, 2, 0)))
                    img_pil_trans = transform_func.rotate(img_pil, angle)
                    # img_pil_trans.save(join(mutual_helper.config.tmp_dir, str(i) + "_img_12.png"))
                    img_tensor = pre_transform(img_pil_trans).unsqueeze(0).to(mutual_helper.equipment)
                    prob = infer_img(mutual_helper, model_seg_coarse, model_cls, img_tensor)
                    res = prob.detach().cpu().squeeze().numpy() * 255.
                    prob_pil = Image.fromarray(res.astype(np.uint8))
                    # prob_pil.save(join(mutual_helper.config.tmp_dir, str(i) + "_probs_02.png"))
                    prob_pil_trans = transform_func.rotate(prob_pil, 0 - angle)
                    # prob_pil_trans.save(join(mutual_helper.config.tmp_dir, str(i) + "_probs_12.png"))
                    images_one_trans.append(np.array(prob_pil_trans))

                images_one_trans = np.stack(images_one_trans)
                probs = np.sum(images_one_trans, axis=0) / len(images_one_trans)
                probs = probs / 255.
                pred_label.append(probs)
                image_names.append(image_name[ii])
                image_sizes.append(image_size[ii])

                img_tmp = images[ii].detach().cpu().numpy().astype(np.uint8)
                img_pil = Image.fromarray(np.transpose(img_tmp, (1, 2, 0)))
                img_pil.save(join(mutual_helper.config.tmp_dir, image_name[ii] + ".png"))
                visualize(np.expand_dims(probs, axis=-1),
                          join(mutual_helper.config.tmp_dir, image_name[ii] + "_probs"))

    pred_label = np.array(pred_label)
    pred_label = np.expand_dims(pred_label, axis=1).astype(np.float32)
    mutual_helper.evaluation_seg_2018(pred_label, image_names, image_sizes, ac_iter, threshold=0.5)


if __name__ == '__main__':
    torch.manual_seed(6666)
    torch.cuda.manual_seed(6666)
    random.seed(6666)
    np.random.seed(6666)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    torch.backends.cudnn.benchmark = True  # cudn
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='train_waterfall_cls.txt')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='test not need write', default=False)

    args, extra_args = argparser.parse_known_args()
    file = 'configuration.txt'
    root = '../../log/ISIC2017_isicgen/Integrate_Model_Seg_Ensemble_Fusion/none/budget_1/gen_3_run_gammar_learn_34_alpha_025_auto_aug'
    config = Configurable(join(root, file), extra_args, isTrain=args.train)
    # config = Configurable(args.config_file, extra_args, isTrain=args.train)
    torch.set_num_threads(config.workers + 1)

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config)
