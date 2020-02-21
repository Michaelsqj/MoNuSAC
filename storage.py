# Python STL
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
# PyTorch
import torch
import math
# Local
import utils
import metrics
import torchsnooper


class Meter(object):
    def __init__(self):

        self.phases = ('train', 'val'),
        self.scores = ('loss', 'iou', 'dice', 'acc', 'prec')

        self.store: Dict[str, Dict[str, List[float]]] = {
            'loss': {'train': [], 'val': []},
            'dice': {'train': [], 'val': []},
            'acc': {'train': [], 'val': []}
        }
        self.metrics = {'loss': 0., 'dice': 0., 'acc': 0.}
        self.base_threshold: float = 0.5

        self.epoch_start_time: datetime = datetime.now()
        self.current_phase = 'train'
        self.batch_num = 0

    def on_epoch_begin(self,
                       current_epoch: int,
                       current_phase: str, ):

        self.epoch_start_time: datetime = datetime.now()
        epoch_start_time_string = datetime.strftime(self.epoch_start_time,
                                                    '%I:%M:%S %p')
        logger = logging.getLogger('')
        logger.info(f"Starting epoch: {current_epoch} | "
                    f"phase: {current_phase} | "
                    f"@ {epoch_start_time_string}")
        self.current_phase = current_phase
        self.batch_num = 0

    def on_epoch_close(self):

        # Average over metrics obtained for every batch in the current epoch
        for key in self.metrics.keys():
            self.metrics[key] = float(self.metrics[key]) / float(self.batch_num)

        # Compute time taken to complete the epoch
        epoch_end_time: datetime = datetime.now()
        delta_t: timedelta = epoch_end_time - self.epoch_start_time
        metric_string: str = f""
        for metric_name, metric_value in self.metrics.items():
            metric_string += f"{metric_name}: {metric_value:.4f} | "
        metric_string += f"in {delta_t.seconds}s"

        logger = logging.getLogger('')
        logger.info(f"{metric_string}")

        for key in self.metrics.keys():
            self.store[key][self.current_phase].append(self.metrics[key])

    # @torchsnooper.snoop()
    def on_batch_close(self,
                       loss: torch.Tensor,
                       np_probs: torch.Tensor,
                       targets: torch.Tensor):
        # np_probs N*2*H*W      targets: N*H*W
        # targets = torch.zeros(size=np_probs.shape).scatter_(dim=1, index=targets.unsqueeze(dim=1).long(), value=1)
        np_preds = torch.argmax(np_probs, dim=1).squeeze()
        assert np_preds.shape == targets.shape
        self.batch_num += 1
        if not torch.isnan(loss):
            self.metrics['loss'] += float(loss)

        dice: torch.Tensor = metrics.dice_score(np_preds, targets)
        if not torch.isnan(dice):
            self.metrics['dice'] += float(dice)

        # iou: torch.Tensor = metrics.iou_score(np_preds, targets)
        # if not torch.isnan(iou):
        #     self.metrics['iou'] += float(iou)

        acc: torch.Tensor = metrics.accuracy_score(np_preds, targets)
        if not torch.isnan(acc):
            self.metrics['acc'] += float(acc)

        # prec: torch.Tensor = metrics.precision_score(np_preds, targets)
        # if not torch.isnan(prec):
        #     self.metrics['prec'] += float(prec)
