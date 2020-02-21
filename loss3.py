# PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-3) -> torch.Tensor:
        assert input.dim() == 3
        assert target.dim() == 3
        N = target.size(0)

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
                np_targets: torch.Tensor):
        assert np_targets.dim() == 3
        # nc_targets = F.one_hot(nc_targets, num_classes=5)
        # nc_targets = nc_targets.permute(0, 3, 1, 2)
        # F.cross_entropy can automatically do the one hot for targets
        # https://blog.csdn.net/zhaowangbo/article/details/100039837
        CEloss = self.ce(F.log_softmax(np_logits, dim=1), np_targets.long())
        Dice = self.dice(F.softmax(np_logits, dim=1), np_targets)
        loss = CEloss + Dice
        # logger = logging.getLogger('')
        # logger.info(f'NP_CE{CEloss.item()},  NP_Dice{Dice.item()}')
        return loss, CEloss, Dice


class _NCBranchLoss(nn.Module):
    def __init__(self):
        super(_NCBranchLoss, self).__init__()
        self.dice = DiceCoeff()
        self.ce = torch.nn.NLLLoss()

    def forward(self, nc_logits: torch.Tensor, nc_targets: torch.Tensor):
        nc_targets = nc_targets.squeeze()
        assert nc_targets.dim() == 3
        CEloss = self.ce(F.log_softmax(nc_logits, dim=1), nc_targets.long())
        Dice = self.dice(F.softmax(nc_logits, dim=1), nc_targets)
        loss = CEloss + Dice
        # logger = logging.getLogger('')
        # logger.info(f'NC_CE{CEloss.item()},  NC_Dice{Dice.item()}')
        return loss, CEloss, Dice


class HoverLoss(nn.Module):
    def __init__(self):
        super(HoverLoss, self).__init__()
        self.np_loss = _NPBranchLoss()
        self.nc_loss = _NCBranchLoss()
        # weights=1,2, 1,1, 1,1

    def forward(self, np_logits, np_targets,
                nc_logits, nc_targets):
        # logits N*2*H*W nc_logits N*5*H*W
        # targets N*H*W except hv_targets N*2*H*W  2 Channel: hv_x and hv_y
        assert (np_targets.dim() == 3 and nc_targets.dim() == 3)
        assert (np_logits.dim() == 4 and nc_logits.dim() == 4)
        assert (np_logits.shape[1] == 2 and nc_logits.shape[1] == 5)
        loss_np, CE_np, Dice_np = self.np_loss(np_logits, np_targets)
        loss_nc, CE_nc, Dice_nc = self.nc_loss(nc_logits, nc_targets)
        loss = 5*loss_np + loss_nc
        return loss, loss_np, loss_nc
