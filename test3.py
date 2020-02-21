import os
import argparse

# parser = argparse.ArgumentParser(description='')
# parser.add_argument('-g', '--gpu', default='3')
# args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model3
import utils
import scipy.io as io
import data
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-3) -> torch.Tensor:
        # print(input.shape,target.shape)
        # assert input.dim() == 3
        # assert target.dim() == 3
        input, target = input.unsqueeze(dim=0), target.unsqueeze(dim=0)
        N = input.shape[0]

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = (2 * intersection.sum(dim=1) + smooth) / (input_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
        loss = 1 - loss.mean().squeeze()

        return loss


class DiceCoeff(nn.Module):
    def __init__(self):
        super(DiceCoeff, self).__init__()
        self.dice = DiceLoss()

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor,
                smooth: float = 1e-3) -> torch.Tensor:
        assert targets.dim() == 3
        tmp = torch.zeros(size=inputs.shape).cuda(device)
        targets = tmp.scatter_(dim=1, index=targets.unsqueeze(dim=1).long(), value=1).cuda()
        assert inputs.shape[1] in (2, 5)
        assert targets.shape == inputs.shape
        totalloss = 0
        for c in range(inputs.shape[1]):
            totalloss += self.dice(inputs[:, c, :, :].squeeze(), targets[:, c, :, :].squeeze())
        return totalloss.cuda() / (inputs.shape[1])


class _NPBranchLoss(nn.Module):
    def __init__(self):
        super(_NPBranchLoss, self).__init__()
        self.dice = DiceCoeff()
        self.ce = torch.nn.NLLLoss()

    def forward(self,
                np_logits: torch.Tensor,
                np_targets: torch.Tensor) -> torch.Tensor:
        assert np_targets.dim() == 3
        # nc_targets = F.one_hot(nc_targets, num_classes=5)
        # nc_targets = nc_targets.permute(0, 3, 1, 2)
        # F.cross_entropy can automatically do the one hot for targets
        # https://blog.csdn.net/zhaowangbo/article/details/100039837
        CEloss = self.ce(F.log_softmax(np_logits, dim=1), np_targets.long())
        Dice = self.dice(F.softmax(np_logits, dim=1), np_targets)
        print(f'np_dice:{1 - Dice.item()}, np_ce:{CEloss.item()}')
        loss = CEloss + Dice
        # logger = logging.getLogger('')
        # logger.info(f'NP_CE{CEloss.item()},  NP_Dice{Dice.item()}')
        return loss


class _NCBranchLoss(nn.Module):
    def __init__(self):
        super(_NCBranchLoss, self).__init__()
        self.dice = DiceCoeff()
        self.ce = torch.nn.NLLLoss()

    def forward(self, nc_logits: torch.Tensor, nc_targets: torch.Tensor) -> torch.Tensor:
        assert nc_targets.dim() == 3
        CEloss = self.ce(F.log_softmax(nc_logits, dim=1), nc_targets.long())
        Dice = self.dice(F.softmax(nc_logits, dim=1), nc_targets)
        print(f'nc_dice:{1 - Dice.item()}, nc_ce:{CEloss.item()}')
        loss = CEloss + Dice
        # logger = logging.getLogger('')
        # logger.info(f'NC_CE{CEloss.item()},  NC_Dice{Dice.item()}')
        return loss


class HoverLoss(nn.Module):
    def __init__(self):
        super(HoverLoss, self).__init__()
        self.np_loss = _NPBranchLoss()
        self.nc_loss = _NCBranchLoss()
        # weights=1,2, 1,1, 1,1

    def forward(self, np_logits, np_targets,
                nc_logits, nc_targets) -> torch.Tensor:
        # logits N*2*H*W nc_logits N*5*H*W
        # targets N*H*W except hv_targets N*2*H*W  2 Channel: hv_x and hv_y
        assert (np_targets.dim() == 3 and nc_targets.dim() == 3)
        assert (np_logits.dim() == 4 and nc_logits.dim() == 4)
        assert (np_logits.shape[1] == 2 and nc_logits.shape[1] == 5)
        loss = self.np_loss(np_logits, np_targets) + self.nc_loss(nc_logits, nc_targets)
        return loss


class predict(nn.Module):
    def __init__(self, datafolder, model_save_path):
        super(predict, self).__init__()
        self.batch_size = 1
        self.dataloader = data.provider(data_folder=datafolder, phase='val', input_shape=(256, 256), batch_size=1)
        self.model = model3.HoverNet()
        self.model.load_state_dict(torch.load(model_save_path))
        self.model.eval()
        self.hoverloss = HoverLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def start(self):
        _, (image, target) = enumerate(self.dataloader).__next__()
        image = image.to(self.device).float()
        np_logits, nc_logits = self.model(image)

        np_targets = utils.get_np_targets(target[:, 0, :, :])
        nc_targets = utils.get_nc_targets(target[:, 1, :, :])

        np_targets = np_targets.to(self.device)
        nc_targets = nc_targets.to(self.device)

        loss = self.hoverloss(np_logits, np_targets, nc_logits, nc_targets)
        print(f'loss:{loss.item()}')
        with torch.no_grad():
            np.save('/home/jqshen/MyCode/MyModel/prediction/np_logits.npy', np_logits.cpu().numpy())
            np.save('/home/jqshen/MyCode/MyModel/prediction/np_target.npy', np_targets.cpu().numpy())
            np.save('/home/jqshen/MyCode/MyModel/prediction/nc_logits.npy', nc_logits.cpu().numpy())
            np.save('/home/jqshen/MyCode/MyModel/prediction/nc_target.npy', nc_targets.cpu().numpy())
            np.save('/home/jqshen/MyCode/MyModel/prediction/image.npy', image.cpu().numpy())
