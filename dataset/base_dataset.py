import os
import wandb
import json
from tools.transform import ArrayToTensor, RandomCrop
import torchvision.transforms as transforms
from tools.compact_data import CompactData

import torch.utils.data as data
from rich.progress import track
import numpy as np

import logging
from rich.logging import RichHandler
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")

from rich.traceback import install
install()

USE_CUSTOM_DATAFORMAT = 1

class BaseDataset(data.Dataset):
    def __init__(self, cfg, mode = 'train'):

        self.mode = mode
        self.cfgs = cfg
        self.dataset_path = cfg['dataset']['dataset_folder']
        self.label_folder = cfg['dataset']['label_folder']
        self.gt_index = cfg['dataset']['gt_index']
        self.use_cls = hyper['model']['use_cls']
        self.input_shape = cfg['model']['input_size']

        # 通过读取json文件解析标签
        self.imageName = []
        self.dataDict = {}
        # training data&label
        self.gt = []

        if self.mode == 'train':
            # 数据增强
            self.train_json_path_list = cfg['dataset']['train_json']
            self.train_folder_list = cfg['dataset']['train_folder']
            logger.info('Training dataset folder: {}'.format(self.train_folder_list))

            self.brightness = cfg['augmentation']['brightness']
            self.contrast = cfg['augmentation']['contrast']

            self.bg_prop = cfg['augmentation']['prop_thresh']['bg']
            self.rotation_prop = cfg['augmentation']['prop_thresh']['rotation']
            self.rotation_angle_center = cfg['augmentation']['rotation_angle_center']
            self.rotation_angle_refpt = cfg['augmentation']['rotation_angle_refpt']

            self.erasing_prop = cfg['augmentation']['prop_thresh']['erasing']
            self.bbox_prop = cfg['augmentation']['prop_thresh']['bbox']
            self.use_rotate_noise = cfg['augmentation']['use_noise2d']
            self.noise_angle = cfg['augmentation']['point2d_noise']

            self.transform = transforms.Compose([transforms.ColorJitter(brightness = self.brightness,
                                                                        contrast = self.contrast),
                                                 # transforms.Resize((80, 80)),
                                                 # transforms.ToTensor(),
                                                 ])
        else:
            self.test_folder_list = cfg['dataset']['test_folder']
            self.test_json_path_list = cfg['dataset']['test_json']
            logger.info('Test dataset folder: {}'.format(self.test_folder_list))

            self.transform = transforms.Compose([transforms.Resize((self.input_shape[0], self.input_shape[1])),
                                                 transforms.ToTensor()])
            self.use_custom = False
            logger.info('Data Augmentation: {}'.format('None'))

    def wandb_config(self):
        note = 'lr={0}, gamma={1}, weight_decay={2}, ' \
               'step_size={3}, model_init={4}, batch={5}'.format(self.para_cfgs['train']['lr'],
                                                                 self.para_cfgs['train']['gamma'],
                                                                 self.para_cfgs['train']['weight_decay'],
                                                                 self.para_cfgs['train']['step_size'],
                                                                 self.para_cfgs['model']['init'],
                                                                 self.para_cfgs['train']['batch_size'])

        hyperparameter_defaults = dict(epoch = self.para_cfgs['train']['epochs'],
                                       optimizer = self.para_cfgs['train']['optimizer'],
                                       loss = self.para_cfgs['train']['loss_function'])

        wandb.init(project = self.cfgs['wandb']['project_name'],
                   config = hyperparameter_defaults,
                   name = self.cfgs['wandb']['name'],
                   tags = self.cfgs['wandb']['tags'],
                   notes = note)

    def process_bg(self):
        bg_folder = os.path.join(self.dataset_path, self.bg_folder)
        self.bg_list = os.listdir(bg_folder)

    def phrase_cls(self):
        logger.critical('Start phrasing classification image...')
        cls_images = os.listdir(self.cls_folder)
        logger.critical('Classification image number: {}'.format(len(cls_images)))

        for t in range(1):
            tmp = np.zeros((len(cls_images), 194))
            for index in track(range(len(cls_images))):
                i = cls_images[index]
                i_name = i.split('.')[0]
                label = [-100] * 194

                tmp_cls = int(i_name.split('_')[1]) - 1
                cls_label = label
                cls_label[1] = tmp_cls
                self.imageName.append(os.path.join(self.cls_folder, i))
                tmp[index] = np.array([cls_label])

            self.dataDict['ground_truth'] = np.r_[self.dataDict['ground_truth'], tmp]

    def update_data_dict(self, cData):
        for attr in cData.attrs:
            value = cData.get_attr(attr)
            if attr in self.dataDict.keys():
                self.dataDict[attr] = np.r_[self.dataDict[attr], value]
            else:
                self.dataDict[attr] = value

    def phrase_json(self, image_folder, json_path):
        data = os.path.join(image_folder)

        logger.info('Read json file: {} ...'.format(json_path))
        if USE_CUSTOM_DATAFORMAT:
            cData = CompactData()
            cData.open(json_path, "r")
            for i in range(0, len(cData.keys)):
                cData.keys[i] = os.path.join(data, cData.keys[i] + '.png')
            self.imageName.extend(cData.keys)
            self.update_data_dict(cData)

        else:
            dataDict = json.load(open(json_path))
            for tmpName in track(dataDict.keys()):
                self.imageName.append(os.path.join(data, tmpName + '.png'))
            self.dataDict.update(dataDict)

    def make_dataset(self):

        # 图像的index要和label相同，利用png,txt的文件名是相同的
        logger.info('Start making {} dataloader...'.format(self.mode))

        for i in track(range(len(self.imageName))):
            fileName = os.path.split(self.imageName[i])[-1].split('.')[0]
            if USE_CUSTOM_DATAFORMAT:
                tmpGT = self.dataDict['ground_truth'][i]
            else:
                tmpGT = self.dataDict[fileName]['ground_truth']

            self.gt.append(tmpGT) # list(map(float, tmpGT.split(' ')[:-1]))  # 提出换行符,空字符
