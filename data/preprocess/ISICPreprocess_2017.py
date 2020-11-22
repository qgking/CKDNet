import numpy as np
from tqdm import tqdm
from skimage import transform as sktransform
import os
from skimage import io
import inspect


def mkdir_if_not_exist(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


curr_filename = inspect.getfile(inspect.currentframe())
root_dir = "../../../medical_data/ISIC_2017_Skin_Lesion"
submission_dir = os.path.join(root_dir, 'submissions')

dir_to_make = [submission_dir]
mkdir_if_not_exist(dir_list=dir_to_make)
mkdir_if_not_exist(dir_list=dir_to_make)

ISIC2017_dir = root_dir
data_dir = os.path.join(ISIC2017_dir)
mkdir_if_not_exist(dir_list=[data_dir])

cached_data_dir = os.path.join(ISIC2017_dir, 'cache')

mkdir_if_not_exist(dir_list=[cached_data_dir])

task1_img = 'ISIC-2017_Training_Data'
task1_gt = 'ISIC-2017_Training_Part1_GroundTruth'
task1_validation_img = 'ISIC-2017_Validation_Data'
task1_vali_gt = 'ISIC-2017_Validation_Part1_GroundTruth'
task1_test_img = 'ISIC-2017_Test_v2_Data'
task1_test_gt = 'ISIC-2017_Test_v2_Part1_GroundTruth'

task3_img = task1_img
# task3_img = 'ISIC-2017_Training_Data_Part3'
task3_gt = 'ISIC-2017_Training_Part3_GroundTruth'
task3_validation_img = task1_validation_img
task3_vali_gt = 'ISIC-2017_Validation_Part3_GroundTruth'
task3_test_img = task1_test_img
task3_test_gt = 'ISIC-2017_Test_v2_Part3_GroundTruth'

melanoma = 0  # Melanoma
seborrheic_keratosis = 1  # Melanocytic nevus

classes = [melanoma, seborrheic_keratosis]
class_names = ['melanoma', 'seborrheic_keratosis']

task1_img_dir = os.path.join(data_dir, task1_img)
task1_validation_img_dir = os.path.join(data_dir, task1_validation_img)
task1_test_img_dir = os.path.join(data_dir, task1_test_img)

task3_img_dir = os.path.join(data_dir, task3_img)
task3_validation_img_dir = os.path.join(data_dir, task3_validation_img)
task3_test_img_dir = os.path.join(data_dir, task3_test_img)

task1_gt_dir = os.path.join(data_dir, task1_gt)
task1_vali_gt_dir = os.path.join(data_dir, task1_vali_gt)
task1_test_gt_dir = os.path.join(data_dir, task1_test_gt)

task3_gt_dir = os.path.join(data_dir, task3_gt)
task3_vali_gt_dir = os.path.join(data_dir, task3_vali_gt)
task3_test_gt_dir = os.path.join(data_dir, task3_test_gt)

task1_image_ids = list()
if os.path.isdir(task1_img_dir):
    task1_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task1_img_dir)
                       if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]
    task1_image_ids.sort()

task1_validation_image_ids = list()
if os.path.isdir(task1_validation_img_dir):
    task1_validation_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task1_validation_img_dir)
                                  if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]
    task1_validation_image_ids.sort()

task1_test_image_ids = list()
if os.path.isdir(task1_test_img_dir):
    task1_test_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task1_test_img_dir)
                            if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]
    task1_test_image_ids.sort()

task3_image_ids = list()
if os.path.isdir(task3_img_dir):
    task3_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task3_img_dir)
                       if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]

    task3_image_ids.sort()

task3_validation_image_ids = list()
if os.path.isdir(task3_validation_img_dir):
    task3_validation_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task3_validation_img_dir)
                                  if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]
    task3_image_ids.sort()

task3_test_image_ids = list()
if os.path.isdir(task3_test_img_dir):
    task3_test_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task3_test_img_dir)
                            if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]
    task3_test_image_ids.sort()

task3_gt_fname = 'ISIC-2017_Training_Part3_GroundTruth.csv' if task3_img == 'ISIC-2017_Training_Data' else 'ISIC-2017_Training_Part3_GroundTruth_add.csv'
task3_vali_gt_fname = 'ISIC-2017_Validation_Part3_GroundTruth.csv'
task3_test_gt_fname = 'ISIC-2017_Test_v2_Part3_GroundTruth.csv'

task1_images_npy_prefix = 'task_images'
task1_validation_images_npy_prefix = 'task_validation_images'
task1_test_images_npy_prefix = 'task_test_images'

task3_images_npy_prefix = task1_images_npy_prefix if task3_img == 'ISIC-2017_Training_Data' else 'task3_images'
task3_validation_images_npy_prefix = task1_validation_images_npy_prefix
task3_test_images_npy_prefix = task1_test_images_npy_prefix


def load_image_by_id(image_id, fname_fn, from_dir, output_size=None, return_size=False):
    img_fnames = fname_fn(image_id)
    if isinstance(img_fnames, str):
        img_fnames = [img_fnames, ]

    assert isinstance(img_fnames, tuple) or isinstance(img_fnames, list)

    images = []
    image_sizes = []
    for img_fname in img_fnames:
        img_fname = os.path.join(from_dir, img_fname)
        if not os.path.exists(img_fname):
            raise FileNotFoundError('img %s not found' % img_fname)

        image = io.imread(img_fname)

        image_sizes.append(np.asarray(image.shape[:2]))

        if output_size:
            image = sktransform.resize(image, (output_size, output_size),
                                       order=1, mode='constant',
                                       cval=0, clip=True,
                                       preserve_range=True,
                                       anti_aliasing=True)
        image = image.astype(np.uint8)
        # else:
        #     image = Image.open(img_fname)
        #     save_dir = './tmp'
        #     if not isdir(save_dir):
        #         makedirs(save_dir)
        #     # visualize(np.asarray(image),
        #     #           join(save_dir, os.path.basename(img_fname)[:-4] + "_1"))
        #     image_sizes.append(np.asarray(image.size))
        #     if output_size:
        #         image = transform(image)
        #     image = np.asarray(image)
        #     # visualize(image,
        #     #           join(save_dir, os.path.basename(img_fname)[:-4] + "_2"))
        images.append(image)

    if return_size:
        if len(images) == 1:
            return images[0], image_sizes[0]
        else:
            return np.stack(images, axis=-1), image_sizes

    if len(images) == 1:
        return images[0]
    else:
        return np.stack(images, axis=-1)  # masks


def load_images(image_ids, from_dir, output_size=None, fname_fn=None, verbose=True, return_size=False):
    images = []

    if verbose:
        print('loading images from', from_dir)

    if return_size:
        image_sizes = []
        for image_id in tqdm(image_ids):
            image, image_size = load_image_by_id(image_id,
                                                 from_dir=from_dir,
                                                 output_size=output_size,
                                                 fname_fn=fname_fn,
                                                 return_size=True)
            images.append(image)
            image_sizes.append(image_size)

        return images, image_sizes


    else:
        for image_id in tqdm(image_ids):
            image = load_image_by_id(image_id,
                                     from_dir=from_dir,
                                     output_size=output_size,
                                     fname_fn=fname_fn)
            images.append(image)

        return images


def load_task1_training_images(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    images_npy_filename = os.path.join(cached_data_dir, '%s%s.npy' % (task1_images_npy_prefix, suffix))
    print(images_npy_filename)
    if os.path.exists(images_npy_filename):
        images = np.load(images_npy_filename)
    else:
        images = load_images(image_ids=task1_image_ids,
                             from_dir=task1_img_dir,
                             output_size=output_size,
                             fname_fn=lambda x: '%s.jpg' % x)
        images = np.stack(images).astype(np.uint8)
        np.save(images_npy_filename, images)
    return images


def load_task1_validation_images(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    images_npy_filename = os.path.join(cached_data_dir, '%s%s.npy' % (task1_validation_images_npy_prefix, suffix))
    image_sizes_npy_filename = os.path.join(cached_data_dir,
                                            '%s_sizes%s.npy' % (task1_validation_images_npy_prefix, suffix))

    if os.path.exists(images_npy_filename) and os.path.exists(image_sizes_npy_filename):

        images = np.load(images_npy_filename)
        image_sizes = np.load(image_sizes_npy_filename)

    else:
        images, image_sizes = load_images(image_ids=task1_validation_image_ids,
                                          from_dir=task1_validation_img_dir,
                                          output_size=output_size,
                                          fname_fn=lambda x: '%s.jpg' % x, return_size=True)
        images = np.stack(images).astype(np.uint8)
        image_sizes = np.stack(image_sizes)

        np.save(images_npy_filename, images)
        np.save(image_sizes_npy_filename, image_sizes)

    return images, image_sizes


def load_task1_test_images(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    images_npy_filename = os.path.join(cached_data_dir, '%s%s.npy' % (task1_test_images_npy_prefix, suffix))
    image_sizes_npy_filename = os.path.join(cached_data_dir, '%s_sizes%s.npy' % (task1_test_images_npy_prefix, suffix))
    print('load ' + images_npy_filename)
    if os.path.exists(images_npy_filename) and os.path.exists(image_sizes_npy_filename):

        images = np.load(images_npy_filename)
        image_sizes = np.load(image_sizes_npy_filename)
    else:
        images, image_sizes = load_images(image_ids=task1_test_image_ids,
                                          from_dir=task1_test_img_dir,
                                          output_size=output_size,
                                          fname_fn=lambda x: '%s.jpg' % x, return_size=True)
        images = np.stack(images).astype(np.uint8)
        image_sizes = np.stack(image_sizes)
        np.save(images_npy_filename, images)
        np.save(image_sizes_npy_filename, image_sizes)

    return images, image_sizes


def load_task3_training_images(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    images_npy_filename = os.path.join(cached_data_dir, '%s%s.npy' % (task3_images_npy_prefix, suffix))
    print("load " + images_npy_filename)
    if os.path.exists(images_npy_filename):
        images = np.load(images_npy_filename)
    else:
        images = load_images(image_ids=task3_image_ids,
                             from_dir=task3_img_dir,
                             output_size=output_size,
                             fname_fn=lambda x: '%s.jpg' % x)
        images = np.stack(images).astype(np.uint8)
        np.save(images_npy_filename, images)
    return images


def load_task3_validation_images(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    images_npy_filename = os.path.join(cached_data_dir, '%s%s.npy' % (task3_validation_images_npy_prefix, suffix))
    print('load ' + images_npy_filename)
    if os.path.exists(images_npy_filename):
        images = np.load(images_npy_filename)
    else:
        images = load_images(image_ids=task3_validation_image_ids,
                             from_dir=task3_validation_img_dir,
                             output_size=output_size,
                             fname_fn=lambda x: '%s.jpg' % x)
        images = np.stack(images).astype(np.uint8)
        np.save(images_npy_filename, images)
    return images


def load_task3_test_images(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    images_npy_filename = os.path.join(cached_data_dir, '%s%s.npy' % (task3_test_images_npy_prefix, suffix))

    if os.path.exists(images_npy_filename):

        images = np.load(images_npy_filename)

    else:

        images = load_images(image_ids=task3_test_image_ids,
                             from_dir=task3_test_img_dir,
                             output_size=output_size,
                             fname_fn=lambda x: '%s.jpg' % x)
        images = np.stack(images).astype(np.uint8)
        np.save(images_npy_filename, images)

    return images


def load_task1_training_masks(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    npy_filename = os.path.join(cached_data_dir, 'task_masks%s.npy' % suffix)
    if os.path.exists(npy_filename):
        masks = np.load(npy_filename)
    else:
        masks = load_images(image_ids=task1_image_ids,
                            from_dir=task1_gt_dir,
                            output_size=output_size,
                            fname_fn=lambda x: '%s_segmentation.png' % x)
        masks = np.stack(masks)
        np.save(npy_filename, masks)
    return masks


def load_task1_vali_masks(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    npy_filename = os.path.join(cached_data_dir, 'task_vali_masks%s.npy' % suffix)
    if os.path.exists(npy_filename):
        masks = np.load(npy_filename)
    else:
        masks = load_images(image_ids=task1_validation_image_ids,
                            from_dir=task1_vali_gt_dir,
                            output_size=output_size,
                            fname_fn=lambda x: '%s_segmentation.png' % x)
        masks = np.stack(masks)
        np.save(npy_filename, masks)
    return masks


def load_task1_test_masks(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    npy_filename = os.path.join(cached_data_dir, 'task_test_masks%s.npy' % suffix)
    if os.path.exists(npy_filename):
        masks = np.load(npy_filename)
    else:
        masks = load_images(image_ids=task1_test_image_ids,
                            from_dir=task1_test_gt_dir,
                            output_size=output_size,
                            fname_fn=lambda x: '%s_segmentation.png' % x)
        masks = np.stack(masks)
        np.save(npy_filename, masks)
    return masks


def load_task3_training_labels():
    # image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
    labels = []
    with open(os.path.join(task3_gt_dir, task3_gt_fname), 'r') as f:
        for i, line in tqdm(enumerate(f.readlines()[1:])):
            fields = line.strip().split(',')
            labels.append([eval(field) for field in fields[1:]])
        labels = np.stack(labels, axis=0)
        labels = np.concatenate([labels, np.expand_dims(1 - np.sum(labels, axis=1), axis=1)], axis=1)
    return labels


def load_task3_vali_labels():
    labels = []
    with open(os.path.join(task3_vali_gt_dir, task3_vali_gt_fname), 'r') as f:
        for i, line in tqdm(enumerate(f.readlines()[1:])):
            fields = line.strip().split(',')
            labels.append([eval(field) for field in fields[1:]])
        labels = np.stack(labels, axis=0)
        labels = np.concatenate([labels, np.expand_dims(1 - np.sum(labels, axis=1), axis=1)], axis=1)
    return labels


def load_task3_test_labels():
    labels = []
    with open(os.path.join(task3_test_gt_dir, task3_test_gt_fname), 'r') as f:
        for i, line in tqdm(enumerate(f.readlines()[1:])):
            fields = line.strip().split(',')
            labels.append([eval(field) for field in fields[1:]])
        labels = np.stack(labels, axis=0)
        labels = np.concatenate([labels, np.expand_dims(1 - np.sum(labels, axis=1), axis=1)], axis=1)
    return labels


def load_training_data(task_idx=1,
                       output_size=None, ):
    assert isinstance(task_idx, int) and 0 < task_idx <= 3
    x = load_task1_training_images(output_size=output_size)
    x = np.transpose(x, (0, 3, 1, 2))
    y = load_task1_training_masks(output_size=output_size)
    y = np.where(y > 0, 1, 0)
    y = np.expand_dims(y, axis=1)
    z = load_task3_training_labels()
    z = np.argmax(z, axis=1)
    # task1_output_map = lambda x: 1 if x == 0 else 0
    # task2_output_map = lambda x: 1 if x == 1 else 0
    # y = np.array(list(map(task1_output_map, y)))
    return x, y, z


def load_validation_data(task_idx=1, output_size=None):
    assert isinstance(task_idx, int) and 0 < task_idx <= 3
    images, image_sizes = load_task1_validation_images(output_size=output_size)
    images = np.transpose(images, (0, 3, 1, 2))
    y = load_task1_vali_masks(output_size=output_size)
    y = np.where(y > 0, 1, 0)
    y = np.expand_dims(y, axis=1)
    z = load_task3_vali_labels()
    z = np.argmax(z, axis=1)
    # task1_output_map = lambda x: 1 if x == 0 else 0
    # task2_output_map = lambda x: 1 if x == 1 else 0
    # y = np.array(list(map(task1_output_map, y)))
    return images, y, z


def load_test_data(task_idx=1, output_size=None):
    assert isinstance(task_idx, int) and 0 < task_idx <= 3
    images, image_sizes = load_task1_test_images(output_size=output_size)
    images = np.transpose(images, (0, 3, 1, 2))
    y = load_task1_test_masks(output_size=output_size)
    y = np.where(y > 0, 1, 0)
    y = np.expand_dims(y, axis=1)
    z = load_task3_test_labels()
    z = np.argmax(z, axis=1)
    # task1_output_map = lambda x: 1 if x == 0 else 0
    # task2_output_map = lambda x: 1 if x == 1 else 0
    # y = np.array(list(map(task1_output_map, y)))
    return images, y, z, task1_test_image_ids, image_sizes


if __name__ == '__main__':
    load_training_data(task_idx=1, output_size=224)
    # load_training_data(task_idx=3, output_size=320)
    load_validation_data(task_idx=1, output_size=224)
    load_test_data(task_idx=1, output_size=224)
