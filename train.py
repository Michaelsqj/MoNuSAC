import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-g', '--gpu', default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from typing import Tuple
import logging
from logging.config import dictConfig
import matplotlib.pyplot as plt
import model
import loss
import data
import utils
import storage
import torch
from tqdm import tqdm
import numpy as np
import model2

data_dir = '/home/qijia/MyCode/MyModel/dataset'
model_save_path = '/home/qijia/MyCode/MyModel/checkpoints'
log_path = '/home/qijia/MyCode/MyModel/hover.log'
metric_store_path = '/home/qijia/MyCode/MyModel/hover.npy'
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
                 val_freq, model_save_path, data_dir: str,
                 input_shape: Tuple[int, int] = (270, 270), checkpoint_path: str = None):

        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch

        self.val_freq: int = val_freq

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = model
        self.net = self.net.to(self.device)
        self.checkpoint_path = checkpoint_path
        self.model_save_path: str = model_save_path

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=3,
                                                                    verbose=True, min_lr=3e-6)
        self.hoverloss = loss.HoverLoss()
        self.best_loss: float = float("inf")  # Very high

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
                batch_size=self.batch_size,
                input_shape=input_shape
            )

        }

        self.meter = storage.Meter()

    def iterate(self,
                epoch: int,
                phase: str) -> float:

        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]

        self.meter.on_epoch_begin(epoch, phase)
        for itr, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):

            images = images.to(self.device).float()
            targets = targets.to(self.device).float()

            np_logits, hv_logits, nc_logits = self.net(images)

            np_targets = utils.get_np_targets(targets[:, 0, :, :])
            hv_targets = utils.get_hv_targets(targets[:, 0, :, :])
            nc_targets = utils.get_nc_targets(targets[:, 1, :, :])
            loss = self.hoverloss(np_logits, np_targets,
                                  hv_logits, hv_targets,
                                  nc_logits, nc_targets)
            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # Update metrics for this batch
            with torch.no_grad():
                loss = loss.detach().cpu()
                logits = logits.detach().cpu()
                # ===ON_BATCH_CLOSE===
                self.meter.on_batch_close(loss=loss, np_probs=np_logits, targets=targets)

        # Collect loss & scores
        self.meter.on_epoch_close()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.meter.store['loss'][phase][-1]

    def start(self):

        for epoch in range(self.max_epoch):

            self.iterate(epoch, "train")

            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            if epoch % self.val_freq == 0:
                val_loss = self.iterate(epoch, "val")

                self.scheduler.step(val_loss)

                if val_loss < self.best_loss:
                    logger = logging.getLogger('')
                    logger.info("****** New optimal found, saving state ******")
                    state["best_loss"] = self.best_loss = val_loss
                    torch.save(state, self.model_save_path)
            print()
        np.save(metric_store_path, self.meter.store)


if __name__ == "__main__":
    model_trainer = Trainer(model=model2.HoverNet(), batch_size=1, max_epoch=50, lr=0.01,
                            val_freq=10, model_save_path=model_save_path, data_dir=data_dir,
                            input_shape=(256, 256))
    model_trainer.start()


    def metric_plot(name, scores):
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(scores["train"])),
                 scores["train"],
                 label=f'train {name}')
        plt.plot(range(len(scores["val"])),
                 scores["val"],
                 label=f'val {name}')
        plt.title(f'{name} plot')
        plt.xlabel('Epoch')
        plt.ylabel(f'{name}')
        plt.legend()
        plt.savefig(f'/home/qijia/MyCode/MyModel/{name}.jpg')


    for metric_name, metric_values in model_trainer.meter.store.items():
        metric_plot(metric_name, metric_values)
'''
model_trainer.meter
{'loss': {('train', 'val'): []}, 'iou': {('train', 'val'): []}, 
 'dice': {('train', 'val'): []}, 'acc': {('train', 'val'): []}, 
 'prec': {('train', 'val'): []}}
'''
