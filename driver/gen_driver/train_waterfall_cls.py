import matplotlib

matplotlib.use("Agg")
import sys

sys.path.extend(["../../", "../", "./"])
from commons.utils import *
import torch
from torch.cuda import empty_cache
import random
from driver.gen_driver.ConfigGEN import Configurable
from module.Critierion import DC_and_Focal_loss
from torch.nn import CrossEntropyLoss
from driver.gen_driver.train_waterfall_helper import MUTUALHelper
import configparser
from driver import gt_transform, train_transforms
from models import MODELS
from models.seg_models.deeplab import DeepLab_Aux
from commons.evaluation import correct_predictions, accuracy_check
from driver import OPTIM
from driver import std, mean
from data import ISIC2018, ISIC2017
from torch.nn.utils import clip_grad_norm_

config = configparser.ConfigParser()
import argparse


def main(config):
    model_seg_coarse = DeepLab_Aux(num_classes=1, return_features=True)
    model_cls = MODELS[config.model](backbone=config.backbone, num_classes=config.cls_classes)

    weights = {
        ISIC2017: [374, 254, 1372],
        ISIC2018: [1113, 6705, 514, 327, 1099, 115, 142]
    }
    training_stats = weights[config.data_name]
    weights = np.divide(1, training_stats, dtype='float32') * 100
    criterion = {
        # 'cls_loss': CrossEntropyLoss(),
        'cls_loss': CrossEntropyLoss(
            weight=torch.tensor(weights)),
        'seg_loss': DC_and_Focal_loss(),
    }
    mutual_helper = MUTUALHelper(model_cls, criterion,
                                 config)
    mutual_helper.move_to_cuda()
    model_seg_coarse.to(mutual_helper.equipment)
    optimizer_seg = OPTIM[mutual_helper.config.learning_algorithm](model_seg_coarse.parameters(),
                                                                   lr=mutual_helper.config.learning_rate,
                                                                   weight_decay=1e-4)

    optimizer_cls = mutual_helper.reset_optim(branch='cls')
    print("data name ", mutual_helper.config.data_name)

    train_loader_mtl_normal, _, _, _ = mutual_helper.get_data_loader(
        global_transform=train_transforms, gt_transform=gt_transform,
        default_label_size=mutual_helper.config.default_seg_label_size,
        train_batch_size=mutual_helper.config.train_seg_batch_size,
        test_batch_size=mutual_helper.config.test_seg_batch_size, task=1)
    _, vali_loader_mtl_normal, _, _ = mutual_helper.get_data_loader(
        global_transform=train_transforms, gt_transform=gt_transform,
        default_label_size=mutual_helper.config.default_cls_label_size,
        train_batch_size=mutual_helper.config.train_cls_batch_size,
        test_batch_size=mutual_helper.config.test_cls_batch_size, task=3)
    train_loader_mtl, \
    vali_loader_mtl, \
    data_sets_mtl, \
    index_split_mtl = mutual_helper.get_aug_data_loader(branch='cls',
                                                        online_aug='online',
                                                        vali_aug='normal',
                                                        default_label_size=mutual_helper.config.default_cls_label_size,
                                                        train_batch_size=mutual_helper.config.train_cls_batch_size,
                                                        test_batch_size=mutual_helper.config.test_cls_batch_size)
    try:
        model_seg_coarse = mutual_helper.load_pretrained_coarse_seg_model(model_seg_coarse)
    except FileExistsError as e:
        for epoch in range(30):
            train_seg_coarse(mutual_helper, model_seg_coarse, train_loader_mtl_normal, optimizer_seg, epoch)
            mutual_helper.log.flush()
        save_model_path = join(mutual_helper.config.save_model_path, "coarse_seg_model.pt")
        print("saved %s" % (save_model_path))
        torch.save(model_seg_coarse.state_dict(), save_model_path)
    model_seg_coarse.eval()
    decay_epoch = [30, 60, 90, 100, 140]
    for nb_acl_iter in range(mutual_helper.config.nb_active_learning_iter):
        best_cls_acc = 0
        decay_next = 0
        decay_e = decay_epoch[decay_next]
        bad_step = 0
        empty_cache()
        for epoch in range(mutual_helper.config.epochs):
            train_critics = {
                'train/cls_loss': 0,
                'train/cls_acc': 0,
                'train/seg_loss': 0,
                'train/seg_acc': 0,
            }
            train_critics_cls = train_cls(mutual_helper, model_seg_coarse, train_loader_mtl, optimizer_cls, epoch)
            train_critics.update(train_critics_cls)
            # validation
            vali_critics_cls = test_cls(mutual_helper, model_seg_coarse, vali_loader_mtl_normal, epoch)
            # test_cls(mutual_helper, model_seg_coarse, vali_loader_mtl, epoch)
            vali_critics = {
                'vali/cls_loss': 0,
                'vali/cls_acc': 0,
                'vali/seg_loss': 0,
                'vali/seg_acc': 0,
            }
            vali_critics.update(vali_critics_cls)
            mutual_helper.log.flush()
            mutual_helper.plot_vali_loss(nb_acl_iter, epoch, vali_critics)
            mutual_helper.plot_train_loss(nb_acl_iter, epoch, train_critics)
            if vali_critics['vali/cls_acc'] >= best_cls_acc:
                print(" * Best vali cls acc: history = %.4f, current = %.4f" % (
                    best_cls_acc, vali_critics['vali/cls_acc']))
                best_cls_acc = vali_critics['vali/cls_acc']
                mutual_helper.save_best_checkpoint(save_model=True, iter=nb_acl_iter)
                # predict_cls(mutual_helper, test_loader_cls, nb_acl_iter)
            else:
                bad_step += 1
                if bad_step >= 20:
                    bad_step = 0
                    for g in optimizer_cls.param_groups:
                        current_lr = max(g['lr'] * 0.5, mutual_helper.config.min_lrate)
                        print("Decaying the learning ratio to %.8f" % (current_lr))
                        g['lr'] = current_lr

        if mutual_helper.config.load_best_epoch:
            print("\n-----------load best state of model -----------")
            mutual_helper.load_best_state(iter=nb_acl_iter)

        mutual_helper.log.flush()

    print("\n----------- TRAINING FINISHED -----------")
    mutual_helper.summary_writer.close()


def train_cls(mutual_helper, model_seg_coarse, train_loader_cls, optimizer_cls, epoch):
    mutual_helper.model.train()
    loss_cls = Averagvalue()
    acc_cls = Averagvalue()
    optimizer_cls.zero_grad()
    batch_num = int(np.ceil(len(train_loader_cls.dataset) / float(mutual_helper.config.train_cls_batch_size)))
    for i, batch in enumerate(train_loader_cls):
        images, _, labels = mutual_helper.generate_batch(batch)
        images_cls_logits, backbone_out, _, _ = model_seg_coarse(images)
        probs = torch.sigmoid(images_cls_logits)
        # image_grid = make_grid(images, nrow=4, padding=2)
        # prob_grid = make_grid(probs, nrow=4, padding=2)
        # visualize(np.clip(np.transpose(image_grid.detach().cpu().numpy(), (1, 2, 0)) * std + mean, 0, 1),
        #           join(mutual_helper.config.submission_dir,
        #                'image_' + str(ii) + str(sub_iter) + '_batch_origin'))
        # visualize(np.clip(np.transpose(prob_grid.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
        #           join(mutual_helper.config.submission_dir,
        #                'image_' + str(ii) + str(sub_iter)  + '_batch_prob'))
        predictions = mutual_helper.model(images, probs, backbone_out)
        loss = mutual_helper.criterions['cls_loss'](predictions, labels)
        prob = F.softmax(predictions, dim=1)
        _, equals = correct_predictions(prob, labels)
        loss_cls.update(loss.item())
        acc_cls.update(equals / images.size(0))
        loss.backward()
        if (i + 1) % mutual_helper.config.update_every == 0 or i == batch_num - 1:
            clip_grad_norm_(filter(lambda p: p.requires_grad, mutual_helper.model.parameters()), \
                            max_norm=mutual_helper.config.clip)
            optimizer_cls.step()
            optimizer_cls.zero_grad()
    print(
        "[Epoch %d] [%s loss: %f] [%s acc: %f]" % (
            epoch,
            'cls', loss_cls.avg, 'cls', acc_cls.avg
        )
    )
    empty_cache()
    return {
        'train/cls_loss': loss_cls.avg,
        'train/cls_acc': acc_cls.avg
    }


def test_cls(mutual_helper, model_seg_coarse, vali_loader_cls, epoch):
    mutual_helper.model.eval()
    loss_cls = Averagvalue()
    acc_cls = Averagvalue()
    with torch.no_grad():
        for i, batch in enumerate(vali_loader_cls):
            images_cls, _, labels_cls = mutual_helper.generate_batch(batch)
            images_cls_logits, backbone_out, _, _ = model_seg_coarse(images_cls)
            probs = torch.sigmoid(images_cls_logits)
            # image_grid = make_grid(images_cls, nrow=4, padding=2)
            # prob_grid = make_grid(probs, nrow=4, padding=2)
            # visualize(np.clip(np.transpose(image_grid.detach().cpu().numpy(), (1, 2, 0)) * std + mean, 0, 1),
            #           join(mutual_helper.config.submission_dir,
            #                'image_' + str(i) + '_batch_origin'))
            # visualize(np.clip(np.transpose(prob_grid.detach().cpu().numpy(), (1, 2, 0)), 0, 1),
            #           join(mutual_helper.config.submission_dir,
            #                'image_' + str(i) + '_batch_prob'))
            predictions_cls = mutual_helper.model(images_cls, probs, backbone_out)
            loss = mutual_helper.criterions['cls_loss'](predictions_cls, labels_cls)
            prob = F.softmax(predictions_cls, dim=1)
            _, equals = correct_predictions(prob, labels_cls)
            loss_cls.update(loss.item())
            acc_cls.update(equals / images_cls.size(0))
        empty_cache()
    info = 'Vali Epoch: [{0}/{1}]'.format(epoch, mutual_helper.config.epochs) + \
           ' Loss {loss:f} '.format(loss=loss_cls.avg) + \
           ' Acc {acc:f} '.format(acc=acc_cls.avg)
    print(info)
    return {
        'vali/cls_loss': loss_cls.avg,
        'vali/cls_acc': acc_cls.avg,
    }


def train_seg_coarse(mutual_helper, model_seg_coarse, train_loader_seg, optimizer_seg, epoch):
    model_seg_coarse.train()
    loss_seg = Averagvalue()
    acc_seg = Averagvalue()
    data_iter_seg = mutual_helper.read_data(train_loader_seg)
    total_iter = len(train_loader_seg)
    for ii in range(total_iter):
        batch_gen_seg = next(data_iter_seg)
        labels = batch_gen_seg['image_segment']
        images = batch_gen_seg['image_patch']
        logits, _, _, _ = model_seg_coarse(images)
        probs = torch.sigmoid(logits)
        loss = mutual_helper.criterions['seg_loss'](probs, labels)
        acc = accuracy_check(probs, labels)
        optimizer_seg.zero_grad()
        loss.backward()
        optimizer_seg.step()
        loss_seg.update(loss.item())
        acc_seg.update(acc)
    empty_cache()
    print(
        "[Epoch %d] [%s loss: %f] [%s acc: %f]" % (
            epoch,
            'seg', loss_seg.avg, 'seg', acc_seg.avg
        )
    )
    return {
        'train/seg_loss': loss_seg.avg,
        'train/seg_acc': acc_seg.avg,
    }


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
    argparser.add_argument('--train', help='test not need write', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args, isTrain=args.train)
    torch.set_num_threads(config.workers + 1)
    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config)
