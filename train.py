import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

import yaml
from core.dataset import  HandPoseDataset, DataLoaderX
from core.select_model import select_model
from core.trainer import Train
import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)

from rich.traceback import install
install()

if __name__ == '__main__':

    cfg_path = 'config/train.yaml'
    cfgs = yaml.load(open(cfg_path), Loader = yaml.FullLoader)
    hyper_para = yaml.load(open(cfgs['model']['hyperPara_path'], 'r'), Loader = yaml.FullLoader)

# =================================================================================================== #
#                                           1. Dataloader                                             #
# =================================================================================================== #

    trainset = HandPoseDataset(cfgs, hyper_para)
    trainloader = DataLoaderX(trainset, batch_size = hyper_para['train']['batch_size'],
                              shuffle = hyper_para['train']['shuffle'],
                              num_workers = hyper_para['train']['num_worker'],
                              pin_memory = True, drop_last = True)

    valset = HandPoseDataset(cfgs, hyper_para, mode = 'val')
    valloader = DataLoaderX(valset, batch_size = cfgs['val']['batch_size'], shuffle = cfgs['val']['shuffle'],
                            num_workers = cfgs['val']['num_worker'])

# =================================================================================================== #
#                                              2. Model                                               #
# =================================================================================================== #

    logger.info('Model init & loading pretrain model...')
    model = select_model(cfgs, hyper_para)

# =================================================================================================== #
#                                              3. Train                                               #
# =================================================================================================== #

    logger.info('Start training...')
    train = Train(cfgs, model, trainloader, valloader)
    train.start_train()