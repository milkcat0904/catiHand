import os
import torch

from torchsummary import summary
from models.modules.baseline_modules import Backbone

import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")

from rich.traceback import install
install()

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class Baseline(Backbone):
    def __init__(self, hyper):
        super(Baseline, self).__init__(hyper)
        if self.point2d_cls_layer == 'fc':
            self.regression2d = self.fc_regression(addon_l = 0, output_dim = self.output_dim['2d'])

        # 卷积cls
        else:
            self.regression12d = self.conv_regression(input_channel = self.input_channel[-1],
                                                      output_channel = self.num_cell,
                                                      output_dim = self.output_dim['2d'])

        self.regression_new_hand = self.fc_classifier(addon_l = 0, output_dim = self.output_dim['new_hand'])
        self.regression_rot_mat = self.fc_classifier(addon_l = 0, output_dim = self.output_dim['rot_mat'])
        self.rot_ratio = hyper['train']['rot_loss_ratio']

        # 参数初始化
        if hyper['model']['resume'] == False:
            logger.critical('Init para...')
            self.init_weights()

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x) # b 96 2 2

        if self.point2d_cls_layer == 'fc':
            x = torch.flatten(x, 1)
            x_2d = self.classifier2d(x)

        else:
            x_2d = self.classifier2d(x)[:,:, 0, 0] # 2, 42, 1, 1 -> 2, 42
            x = torch.flatten(x, 1)

        x_new = self.classifier_new_hand(x) # 2, 63
        if self.rot_ratio == 0:
            out = torch.cat((x_2d, x_new), dim = 1)
        else:
            x_rot_mat = self.classifier_rot_mat(x)
            out = torch.cat((x_2d, x_new, x_rot_mat), dim = 1)
        return out

def make_network(train_cfg, hyper_data):
    input_size = train_cfg['model']['input_size']
    model = Baseline(hyper_data)

    # 可视化
    summary(model, (input_size[2], input_size[1], input_size[0]), device = 'cpu') # (C, H, W)

    # load checkpoint
    if hyper_data['model']['resume']:
        logger.critical('Loading para from {}...'.format(hyper_data['model']['checkpoint']))

        checkpoint = torch.load(os.path.join(hyper_data['model']['model_folder'], hyper_data['model']['checkpoint']))
        pretrained_dict = checkpoint['model']

        # 去除关键词里的 'module.'字符串
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if 'module' in k:
                k = k.replace('module.', '')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    else:
        pass
    return model