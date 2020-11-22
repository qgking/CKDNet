from configparser import ConfigParser
import configparser
import sys, os

sys.path.append('..')


class Configurable(object):
    def __init__(self, config_file, extra_args, isTrain=True):
        config = ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._config = config
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if isTrain:
            config.write(open(self.config_file, 'w'))
        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    # ------------data config reader--------------------
    @property
    def patch_x(self):
        return self._config.getint('Data', 'patch_x')

    @property
    def patch_y(self):
        return self._config.getint('Data', 'patch_y')

    @property
    def data_name(self):
        return self._config.get('Data', 'data_name')


    @property
    def data_branch(self):
        return self._config.get('Data', 'data_branch')
    # ------------save path config reader--------------------

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def submission_dir(self):
        return self._config.get('Save', 'submission_dir')

    @property
    def tmp_dir(self):
        return self._config.get('Save', 'tmp_dir')

    @property
    def tensorboard_dir(self):
        return self._config.get('Save', 'tensorboard_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def log_file(self):
        return self._config.get('Save', 'log_file')

    # ------------Network path config reader--------------------

    @property
    def model(self):
        return self._config.get('Network', 'model')

    @property
    def cls_classes(self):
        return self._config.getint('Network', 'cls_classes')

    @property
    def backbone(self):
        return self._config.get('Network', 'backbone')

    @property
    def cls_branch(self):
        return self._config.get('Network', 'cls_branch')

    @property
    def seg_branch(self):
        return self._config.get('Network', 'seg_branch')

    @property
    def train_branch(self):
        return self._config.get('Network', 'train_branch')

    # ------------Network path config reader--------------------

    @property
    def epochs(self):
        return self._config.getint('Run', 'N_epochs')

    @property
    def train_seg_batch_size(self):
        return self._config.getint('Run', 'train_seg_batch_size')

    @property
    def train_cls_batch_size(self):
        return self._config.getint('Run', 'train_cls_batch_size')

    @property
    def test_cls_batch_size(self):
        return self._config.getint('Run', 'test_cls_batch_size')

    @property
    def test_seg_batch_size(self):
        return self._config.getint('Run', 'test_seg_batch_size')

    @property
    def gpu(self):
        return self._config.getint('Run', 'gpu')

    @property
    def run_num(self):
        return self._config.getint('Run', 'run_num')

    @property
    def load_best_epoch(self):
        return self._config.getboolean('Run', 'load_best_epoch')

    @property
    def printfreq(self):
        return self._config.getint('Run', 'printfreq')

    @property
    def gpu_count(self):
        gpus = self._config.get('Run', 'gpu_count')
        gpus = gpus.split(',')
        return [int(x) for x in gpus]

    @property
    def workers(self):
        return self._config.getint('Run', 'workers')

    @property
    def nb_active_learning_iter(self):
        return self._config.getint('Run', 'nb_active_learning_iter')

    @property
    def default_cls_label_size(self):
        return self._config.getint('Run', 'default_cls_label_size')

    @property
    def default_seg_label_size(self):
        return self._config.getint('Run', 'default_seg_label_size')

    @property
    def label_cls_each(self):
        return self._config.getint('Run', 'label_cls_each')

    # ------------Optimizer path config reader--------------------
    @property
    def learning_algorithm(self):
        return self._config.get('Optimizer', 'learning_algorithm')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def cls_learning_rate(self):
        return self._config.getfloat('Optimizer', 'cls_learning_rate')

    @property
    def seg_learning_rate(self):
        return self._config.getfloat('Optimizer', 'seg_learning_rate')

    @property
    def backbone_learning_rate(self):
        return self._config.getfloat('Optimizer', 'backbone_learning_rate')

    @property
    def min_lrate(self):
        return self._config.getfloat('Optimizer', 'min_lrate')

    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer', 'epsilon')

    @property
    def clip(self):
        return self._config.getfloat('Optimizer', 'clip')

    @property
    def update_every(self):
        return self._config.getint('Run', 'update_every')