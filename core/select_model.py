import torch.backends.cudnn
from models.backbone import baseline

import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")

from rich.traceback import install
install()

def select_model(cfgs, hyper_para):
    logger.info('Model: {}'.format(hyper_para['model']['name']))
    model = eval(hyper_para['model']['name'] + '.make_network')(train_cfg=cfgs,
                                                                hyper_data = hyper_para)
    # cuda, 设备
    deviceString = cfgs['model']['device']
    torch.backends.cudnn.benchmark = True
    torch.device(deviceString)
    return model

