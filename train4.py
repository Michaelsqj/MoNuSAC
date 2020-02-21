import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-g', '--gpu', default='1')
parser.add_argument('-p', '--phase', default='train')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
phase = args.phase
from typing import Tuple
import logging
from logging.config import dictConfig
import matplotlib.pyplot as plt
import loss4
import data
import utils
import storage
import torch
from tqdm import tqdm
import numpy as np
import model4
import torchsnooper
import test4

data_dir = '/home/jqshen/MyCode/MyModel/dataset'
model_save_path = '/home/jqshen/MyCode/MyModel/checkpoints/hover4.tar'
log_path = '/home/jqshen/MyCode/MyModel/hover4.log'
metric_store_path = '/home/jqshen/MyCode/MyModel/hover4.npy'
C_LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'loggers': {
        '': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        },
    },
    'formatters': {
        'console': {
            'format': "%(asctime)s - %(name)-18s - %(levelname)-8s"
                      " - %(message)s",
            'datefmt': '%H:%M:%S'},
        'file': {
            'format': "%(asctime)s :: %(levelname)s :: %(funcName)s in "
                      "%(filename)s (l:%(lineno)d) :: %(message)s",
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'console',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': log_path,  # 这是将普通日志写入到日志文件中的方法，
            'formatter': 'file'
        }
    },
}
dictConfig(C_LOGGING)
logger = logging.getLogger('')


class Trainer(object):

    def __init__(self, model, batch_size: int, max_epoch: int, lr: float,
                 val_freq, model_save_path, data_dir: str, phase: str,
                 input_shape: Tuple[int, int] = (256, 256), checkpoint_path: str = None, continue_train=False):
        self.phase = phase
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch

        # self.val_freq: int = val_freq

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = model
        self.net = self.net.to(self.device)
        self.checkpoint_path = checkpoint_path
        self.model_save_path: str = model_save_path
        if continue_train:
            self.net.load_state_dict(torch.load(self.model_save_path))
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=3,
        #                                                             verbose=True, min_lr=3e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=10, gamma=0.6)
        self.hoverloss = loss4.HoverLoss()
        self.best_loss: float = float("inf")  # Very high
        self.val_freq = val_freq
        self.dataloaders = {
            'train': data.provider(
                data_folder=data_dir,
                phase='train',
                batch_size=self.batch_size,
                input_shape=input_shape
            ),
            'val': data.provider(
                data_folder=data_dir,
                phase='val',
                batch_size=int((self.batch_size) / 4),
                input_shape=input_shape
            )
        }

        self.store = {'train': {'loss': [], 'La': [], 'Lb': []}, 'val': {'loss': [], 'La': [], 'Lb': []}}
        self.epoch_loss = {'loss': [], 'La': [], 'Lb': []}

    # @torchsnooper.snoop()
    def iterate(self,
                epoch: int,
                phase: str):

        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]

        for itr, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):

            images = images.to(self.device).float()
            N = images.shape[0]
            hv_logits = self.net(images)

            hv_targets = utils.get_hv_targets(targets[:, 0, :, :])

            hv_targets = hv_targets.to(self.device)

            loss, La, Lb = self.hoverloss(hv_logits, hv_targets)
            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update metrics for this batch
            with torch.no_grad():
                loss = loss.detach()
                La=La.detach()
                # Lb=Lb.detach()
                self.epoch_loss['loss'].append(loss.item())
                self.epoch_loss['La'].append(La.item())
                # self.epoch_loss['Lb'].append(Lb.item())
        self.store[phase]['loss'].append(sum(self.epoch_loss['loss']) / len(self.epoch_loss['loss']))
        self.store[phase]['La'].append(sum(self.epoch_loss['La']) / len(self.epoch_loss['La']))
        # self.store[phase]['Lb'].append(sum(self.epoch_loss['Lb']) / len(self.epoch_loss['Lb']))
        self.epoch_loss['loss'] = [0]
        self.epoch_loss['La'] = [0]
        # self.epoch_loss['Lb'] = [0]
        return self.store[phase]['loss'][-1]

    def start(self):

        for epoch in range(self.max_epoch):
            self.iterate(epoch, self.phase)
            self.scheduler.step(epoch)
            if epoch % self.val_freq == 0:
                val_loss = self.iterate(epoch, "val")

                # self.scheduler.step(val_loss)

                if val_loss < self.best_loss:
                    logger = logging.getLogger('')
                    logger.info("****** New optimal found, saving state ******")
                    self.best_loss = val_loss
                    torch.save(self.net.state_dict(), self.model_save_path)
            print()
        np.save(metric_store_path, self.store)


if __name__ == "__main__":
    model_trainer = Trainer(model=model4.HoverNet(), batch_size=8, max_epoch=50, lr=1e-4,
                            val_freq=10, model_save_path=model_save_path, data_dir=data_dir,
                            input_shape=(256, 256), phase=phase, continue_train=False)
    model_trainer.start()


    def metric_plot(phase, value):
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(value['loss'])),
                 value['loss'],
                 label=f' {phase}loss')
        plt.plot(range(len(value['La'])),
                 value['La'],
                 label=f' {phase}La')
        # plt.plot(range(len(value['Lb'])),
        #          value['Lb'],
        #          label=f' {phase}Lb')
        # plt.plot(range(len(scores["val"])),
        #          scores["val"],
        #          label=f'val {name}')
        plt.title(f'{phase} plot')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'/home/jqshen/MyCode/MyModel/{phase}_loss.png')


    for phase, value in model_trainer.store.items():
        metric_plot(phase, value)

    tester = test4.predict(datafolder=data_dir, model_save_path=model_save_path)
    tester.start()
'''
model_trainer.meter
{'loss': {('train', 'val'): []}, 'iou': {('train', 'val'): []}, 
 'dice': {('train', 'val'): []}, 'acc': {('train', 'val'): []}, 
 'prec': {('train', 'val'): []}}
'''
