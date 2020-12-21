import os
import json
import torch
import random

import numpy as np
from PIL import Image
import wandb

from rich.progress import track
import torchvision.transforms as transforms
from tools.utils import get_new_hand, get_points, add_rotation_noise, generate_heatmap
import logging
from rich.logging import RichHandler
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")

from rich.traceback import install
install()
import torch.utils.data as data
# from PIL import Image

USE_CUSTOM_DATAFORMAT = 1
from prefetch_generator import BackgroundGenerator

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class BaseDataset(data.Dataset):
    def __init__(self, cfg, mode = 'train'):

        self.mode = mode
        self.cfgs = cfg
        self.dataset_path = cfg['dataset']['dataset_folder']
        self.label_folder = cfg['dataset']['label_folder']
        self.gt_index = cfg['dataset']['gt_index']
        self.input_shape = cfg['model']['input_size']

        # 通过读取json文件解析标签
        self.imageName = []
        self.dataDict = {}
        # training data&label
        self.gt = []

        # wandb
        self.use_wandb = self.cfgs['wandb']['use']
        if self.use_wandb:
            self.wandb_config()

        if self.mode == 'train':
            # 数据增强
            self.train_json_path_list = cfg['dataset']['train_json']
            self.train_folder_list = cfg['dataset']['train_folder']
            logger.info('Training dataset folder: {}'.format(self.train_folder_list))

            self.brightness = cfg['augmentation']['brightness']
            self.contrast = cfg['augmentation']['contrast']

            self.mask_folder = cfg['dataset']['mask_folder']
            self.bg_folder = os.path.join(cfg['dataset']['bg_folder'])
            self.cls_folder = os.path.join(self.dataset_path, cfg['dataset']['cls_folder'])

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
            logger.info('Data Augmentation: brightness {0} | contrast: {1} | rotation_c: {2} rotations_r: {3} thresh: {4} | '
                        'background: mask alpha: {5} thresh: {6} | erasing: thresh: {7} | '
                        'bbox thresh: {8} |'.format(self.brightness, self.contrast,
                                                    self.rotation_angle_center, self.rotation_angle_refpt,
                                                    self.rotation_prop,
                                                    cfg['augmentation']['mask_alpha'], self.bg_prop,
                                                    self.erasing_prop, self.bbox_prop))

            self.use_custom = cfg['augmentation']['use_custom_trans']
            self.custom_transform = transforms.Compose([
                RandomCrop(thresh = self.bbox_prop, crop_size=self.input_shape[0]),
                ArrayToTensor()
            ])

        else:
            self.test_folder_list = cfg['dataset']['test_folder']
            self.test_json_path_list = cfg['dataset']['test_json']
            logger.info('Test dataset folder: {}'.format(self.test_folder_list))

            self.transform = transforms.Compose([transforms.Resize((self.input_shape[0], self.input_shape[1])),
                                                 transforms.ToTensor()])
            self.use_custom = False
            logger.info('Data Augmentation: {}'.format('None'))


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

class HandPoseDataset(BaseDataset):
    def __init__(self, cfg, hyper, mode = 'train'):
        super(HandPoseDataset, self).__init__(cfg, mode)

        if self.mode == 'train':
            self.process_bg()
            for index in range(len(self.train_folder_list)):
                tmpJsonPath = os.path.join(self.label_folder, self.train_json_path_list[index])
                self.phrase_json(os.path.join(self.dataset_path, self.train_folder_list[index]), tmpJsonPath)
                pass

            # 添加分类图像数据
            if self.mode == 'train' and self.use_cls:
                self.phrase_cls()
        else:
            for index in range(len(self.test_folder_list)):
                tmpJsonPath = os.path.join(self.label_folder, self.test_json_path_list[index])
                self.phrase_json(self.test_folder_list[index], tmpJsonPath)

        # self.make_dataset()
        self.make_dataset()
        self.gt = np.array(self.gt)
        logger.info('Data file number: {}'.format(len(self.imageName)))

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


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: image, addon, metadata
        """
        imagePath = self.imageName[index]
        folder = imagePath.split('/')[-2]
        fileName = os.path.split(imagePath)[-1].split('.')[0]
        # imagePath = os.path.join(self.dataDict[fileName]['data_path'], fileName + '.png')

        image = Image.open(imagePath)
        gt = self.gt[index]

        point2d = gt[self.gt_index['point2d'][0]: self.gt_index['point2d'][1]]
        point3d = gt[self.gt_index['point3d'][0]: self.gt_index['point3d'][1]]

        # No Transform
        if self.transform is not None:
            image = self.transform(image)

        # 自定义数据增强
        if self.use_custom == True:
            image = np.array(image)

            if folder == self.mask_folder:
                mask_name = fileName + '_msk' + '.png'
                mask_path = os.path.join(self.dataset_path, self.mask_folder, mask_name)
                mask = np.array(Image.open(mask_path))

                # bg
                choice = random.randint(0, len(self.bg_list) - 1)
                bg_path = os.path.join(self.bg_folder, self.bg_list[choice])
                bg = np.array(Image.open(bg_path))

            else:
                mask = None
                bg = False

            if self.bbox_prop > 0:
                bbox = gt[self.gt_index['bbox'][0]: self.gt_index['bbox'][1]]
                image_dict = {'image': image, 'mask': mask, 'bg': bg, 'point_2d': np.array(point2d), 'bbox': np.array(bbox)}

            else:
                image_dict = {'image': image, 'mask': mask, 'bg': bg, 'point_2d': point2d}
            image = self.custom_transform(image_dict)['image'].float() # 1 64 128

            if self.use_rotate_noise:
                point2d_new = add_rotation_noise(image_dict['point_2d'], self.noise_angle)
            else:
                point2d_new = np.array(image_dict['point_2d'])
            image = torch.unsqueeze(image, dim = 0)/255
            del image_dict
        else:
            image = torch.tensor(np.array(image)).float()
            point2d_new = point2d

        metadata = {}
        point2d_new = torch.tensor(point2d_new)
        point3d = torch.tensor(point3d)

        # 训练label list2tensor
        metadata['bone_len'] = torch.tensor(gt[self.gt_index['bone_len'][0]: self.gt_index['bone_len'][1]])
        metadata['bone_vec'] = torch.tensor(gt[self.gt_index['bone_vec'][0]: self.gt_index['bone_vec'][1]])
        metadata['rot_mat'] = torch.tensor(gt[self.gt_index['rot_mat'][0]: self.gt_index['rot_mat'][1]])

        if point3d[0] < -20:
            metadata['new_hand'] = torch.zeros(63).float()
            metadata['cls'] = point2d_new[1].float() # float64
        else:
            metadata['new_hand'] = get_new_hand(metadata['bone_len'], metadata['bone_vec']).float()
            metadata['cls'] = torch.tensor(-100).float()

        metadata['point2d'] = point2d_new
        metadata['point3d'] = point3d

        metadata['palm2d'] = get_points(metadata['point2d'], point_type = 'palm')
        metadata['tips2d'] = get_points(metadata['point2d'], point_type = 'tips')
        metadata['palm3d'] = get_points(metadata['new_hand'], point_type = 'palm', dim = 3) # 不能用point3d作为输入计算
        metadata['tips3d'] = get_points(metadata['new_hand'], point_type = 'tips', dim = 3)

        #image_show = cv2.cvtColor((image.view(80,80).detach().cpu().numpy()*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        #cv2.imshow("image", image_show)
        #cv2.waitKey(0)

        # addon输入
        metadata['addon_input'] = metadata['new_hand']

        # heatmap
        if self.use_heatmap:
            metadata['heatmap'] = generate_heatmap(point2d_new, heatmap_shape=self.heatmap_size,
                                                   variance = self.heatmap_variance)[0]

        # metadata不能在初始化里定义，多线程调用会出问题
        metadata['file_path'] = imagePath
        return image, metadata

    def __len__(self):
        return len(self.imageName)


