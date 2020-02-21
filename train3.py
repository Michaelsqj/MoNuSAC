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
import loss3
import data
import utils
import storage
import torch
from tqdm import tqdm
import numpy as np
import model3
import torchsnooper
import test3

data_dir = '/home/jqshen/MyCode/MyModel/dataset'
model_save_path = '/home/jqshen/MyCode/MyModel/checkpoints/hover3.tar'
log_path = '/home/jqshen/MyCode/MyModel/hover3.log'
metric_store_path = '/home/jqshen/MyCode/MyModel/hover3.npy'
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
                 input_shape: Tuple[int, int] = (256, 256), checkpoint_path: str = None):
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

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=3,
        #                                                             verbose=True, min_lr=3e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=10, gamma=1)
        self.hoverloss = loss3.HoverLoss()
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

        # self.meter = storage.Meter()
        self.epoch_loss = {'loss': [], 'loss_np': [], 'loss_nc': []}
        self.store = {'train': {'loss': [], 'loss_np': [], 'loss_nc': []},
                      'val': {'loss': [], 'loss_np': [], 'loss_nc': []}}

    # @torchsnooper.snoop()
    def iterate(self,
                epoch: int,
                phase: str):

        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]

        # self.meter.on_epoch_begin(epoch, phase)
        for itr, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):

            images = images.to(self.device).float()
            N = images.shape[0]
            np_logits, nc_logits = self.net(images)

            np_targets = utils.get_np_targets(targets[:, 0, :, :])
            nc_targets = utils.get_nc_targets(targets[:, 1, :, :])

            np_targets = np_targets.to(self.device)
            nc_targets = nc_targets.to(self.device)
            assert np_targets.shape == (N, 256, 256) and nc_targets.shape == (N, 256, 256)

            loss, loss_np, loss_nc = self.hoverloss(np_logits, np_targets, nc_logits, nc_targets)
            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step(epoch)
            # Update metrics for this batch
            with torch.no_grad():
                loss = loss.detach()
                loss_np = loss_np.detach()
                loss_nc = loss_nc.detach()
                self.epoch_loss['loss'].append(loss.item())
                self.epoch_loss['loss_np'].append(loss_np.item())
                self.epoch_loss['loss_nc'].append(loss_nc.item())
                # ===ON_BATCH_CLOSE===
                # self.meter.on_batch_close(loss=loss, np_probs=np_logits, targets=np_targets)
        self.store[phase]['loss'].append(sum(self.epoch_loss['loss']) / len(self.epoch_loss['loss']))
        self.store[phase]['loss_np'].append(sum(self.epoch_loss['loss_np']) / len(self.epoch_loss['loss_np']))
        self.store[phase]['loss_nc'].append(sum(self.epoch_loss['loss_nc']) / len(self.epoch_loss['loss_nc']))
        self.epoch_loss['loss'] = [0]
        self.epoch_loss['loss_np'] = [0]
        self.epoch_loss['loss_nc'] = [0]
        # Collect loss & scores
        # self.meter.on_epoch_close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.store[phase]['loss'][-1]

    def start(self):

        for epoch in range(self.max_epoch):
            self.iterate(epoch, self.phase)

            if epoch % self.val_freq == 0:
                val_loss = self.iterate(epoch, "val")

                # self.scheduler.step(val_loss)

                if val_loss < self.best_loss:
                    logger = logging.getLogger('')
                    logger.info("****** New optimal found, saving state ******")
                    self.best_loss = val_loss
                    # torch.save(state, self.model_save_path)
            print()
        torch.save(self.net.state_dict(), self.model_save_path)


if __name__ == "__main__":
    model_trainer = Trainer(model=model3.HoverNet(), batch_size=8, max_epoch=50, lr=1e-4,
                            val_freq=10, model_save_path=model_save_path, data_dir=data_dir,
                            input_shape=(256, 256), phase=phase)
    model_trainer.start()


    def metric_plot(phase, scores):
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(scores["loss"])),
                 scores["loss"],
                 label='loss')
        plt.plot(range(len(scores["loss_np"])),
                 5*scores["loss_np"],
                 label='loss_np')
        plt.plot(range(len(scores["loss_nc"])),
                 scores["loss_nc"],
                 label='loss_nc')
        plt.title(f'{phase}')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'/home/jqshen/MyCode/MyModel/{phase}.png')


    for phase, scores in model_trainer.store.items():
        metric_plot(phase, scores)

    tester = test3.predict(datafolder=data_dir, model_save_path=model_save_path)
    tester.start()
'''
model_trainer.meter
{'loss': {('train', 'val'): []}, 'iou': {('train', 'val'): []}, 
 'dice': {('train', 'val'): []}, 'acc': {('train', 'val'): []}, 
 'prec': {('train', 'val'): []}}
'''
