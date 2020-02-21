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
                np_targets: torch.Tensor) -> torch.Tensor:
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
        return loss


class _HVBranchLoss(nn.Module):
    def __init__(self):
        super(_HVBranchLoss, self).__init__()
        self.mse1 = torch.nn.MSELoss(size_average=True, reduce=True)
        self.mse2 = torch.nn.MSELoss(size_average=True, reduce=True)

    def forward(self,
                hv_logits: torch.Tensor,
                hv_targets: torch.Tensor):
        hv_logits = torch.tanh(hv_logits)
        La = self.mse1(hv_logits, hv_targets.float())

        h_grads, v_grads = utils.get_gradient_hv(hv_logits, h_ch=0, v_ch=1)
        h_grads_tar, v_grads_tar = utils.get_gradient_hv(hv_targets, h_ch=0, v_ch=1)

        h_grads, v_grads, h_grads_tar, v_grads_tar = h_grads.cuda(), v_grads.cuda(), h_grads_tar.cuda(), v_grads_tar.cuda()
        Lb = self.mse2(h_grads.squeeze(), h_grads_tar.squeeze().float()) + F.mse_loss(v_grads.squeeze(),
                                                                                      v_grads_tar.squeeze().float())

        loss = La
        return loss, La, Lb


class _NCBranchLoss(nn.Module):
    def __init__(self):
        super(_NCBranchLoss, self).__init__()
        self.dice = DiceCoeff()
        self.ce = torch.nn.NLLLoss()

    def forward(self, nc_logits: torch.Tensor, nc_targets: torch.Tensor) -> torch.Tensor:
        nc_targets = nc_targets.squeeze()
        assert nc_targets.dim() == 3
        CEloss = self.ce(F.log_softmax(nc_logits, dim=1), nc_targets.long())
        Dice = self.dice(F.softmax(nc_logits, dim=1), nc_targets)
        loss = CEloss + Dice
        # logger = logging.getLogger('')
        # logger.info(f'NC_CE{CEloss.item()},  NC_Dice{Dice.item()}')
        return loss


class HoverLoss(nn.Module):
    def __init__(self):
        super(HoverLoss, self).__init__()
        self.hv_loss = _HVBranchLoss()
        # weights=1,2, 1,1, 1,1

    def forward(self, hv_logits, hv_targets):
        loss, La, Lb = self.hv_loss(hv_logits, hv_targets)

        return loss, La, Lb
