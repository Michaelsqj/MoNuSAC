# Python STL
from typing import Tuple
# PyTorch
import torch
from torch.nn import functional as F
import numpy as np
import cv2


def get_gradient_hv(logits: torch.Tensor,
                    h_ch: int = 0,
                    v_ch: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get horizontal & vertical gradients

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits from HV branch
    h_ch : int
        Number of horizontal channels
    v_ch : int
        Number of vertical channels

    Returns
    -------
    gradients : Tuple[torch.Tensor, torch.Tensor]
        Horizontal and vertical gradients
    """
    mh = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=1).cuda()
    mv = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=1).cuda()

    hl = logits[:, h_ch, :, :].unsqueeze(dim=1).float()
    vl = logits[:, v_ch, :, :].unsqueeze(dim=1).float()

    assert (mh.dim() == 4 and mv.dim() == 4 and hl.dim() == 4 and vl.dim() == 4)

    dh = F.conv2d(hl, mh, stride=1, padding=1)
    dv = F.conv2d(vl, mv, stride=1, padding=1)

    return dh, dv


def get_np_targets(targets: torch.Tensor) -> torch.Tensor:
    '''

    Parameters
    ----------
    targets N*H*W with the value of {0,...,n} n is the number of cells

    Returns N*H*W with 0:background 1:cells
    -------

    '''
    assert targets.dim() == 3
    targets=targets.double()
    np_targets = targets.where(targets == 0, torch.tensor(1).double())

    return np_targets


def get_hv_targets(targets: torch.Tensor) -> torch.Tensor:
    '''

    Parameters
    ----------
    targets N*H*W with the value of {0,...,n} n is the number of cells

    Returns N*2*H*W
    -------

    '''
    # N: batch size
    N = targets.shape[0]
    hv_targets = np.zeros(shape=(N, 2, targets.shape[1], targets.shape[2]), dtype=float)
    for i in range(N):
        target = targets[i, :, :].squeeze()
        target = torch.Tensor.numpy(target)
        inst_centroid_list = get_inst_centroid(target)  # [(x1,y1),(x2,y2),(x3,y3)....(xn,yn)]
        inst_id_list = list(np.unique(target))
        inst_id_list.pop(0)

        assert len(inst_centroid_list) == len(inst_id_list)
        for id in range(len(inst_id_list)):  # id: instance index from 1~n
            target = target.astype(np.uint8)
            xc, yc = inst_centroid_list[id]
            H, V = np.meshgrid(np.arange(target.shape[0]), np.arange(target.shape[1]))
            xc, yc = int(xc), int(yc)
            tmp_h = H - xc
            tmp_v = V - yc
            tmp_h = np.where(target == inst_id_list[id], tmp_h, 0)
            tmp_v = np.where(target == inst_id_list[id], tmp_v, 0)
            #### rescale to -1~1
            #### horizontal
            maximum = np.max(tmp_h)
            minimum = np.min(tmp_h)
            if maximum > 0 and minimum < 0:
                tmp_h_pos = np.where(tmp_h > 0, tmp_h, 0).astype(float)
                tmp_h_neg = np.where(tmp_h < 0, tmp_h, 0).astype(float)
                tmp_h_pos = tmp_h_pos / maximum
                tmp_h_neg = tmp_h_neg / abs(minimum)
                tmp_h = tmp_h_neg + tmp_h_pos
            elif maximum > 0 and minimum == 0:
                tmp_h_pos = np.where(tmp_h > 0, tmp_h, 0).astype(float)
                tmp_h_pos = tmp_h_pos / maximum
                tmp_h = tmp_h_pos.astype(float)
            elif maximum == 0 and minimum < 0:
                tmp_h_neg = np.where(tmp_h < 0, tmp_h, 0).astype(float)
                tmp_h_neg = tmp_h_neg / abs(minimum)
                tmp_h = tmp_h_neg.astype(float)
            else:
                tmp_h = tmp_h.astype(float)
            #### vertical
            maximum = np.max(tmp_v)
            minimum = np.min(tmp_v)
            if maximum > 0 and minimum < 0:
                tmp_v_pos = np.where(tmp_v > 0, tmp_v, 0).astype(float)
                tmp_v_neg = np.where(tmp_v < 0, tmp_v, 0).astype(float)
                tmp_v_pos = tmp_v_pos / maximum
                tmp_v_neg = tmp_v_neg / abs(minimum)
                tmp_v = tmp_v_neg + tmp_v_pos
            elif maximum > 0 and minimum == 0:
                tmp_v_pos = np.where(tmp_v > 0, tmp_v, 0).astype(float)
                tmp_v_pos = tmp_v_pos / maximum
                tmp_v = tmp_v_pos
            elif maximum == 0 and minimum < 0:
                tmp_v_neg = np.where(tmp_v < 0, tmp_v, 0).astype(float)
                tmp_v_neg = tmp_v_neg / abs(minimum)
                tmp_v = tmp_v_neg
            else:
                tmp_v = tmp_v.astype(float)

            Temp = np.where(target == inst_id_list[id], tmp_h, 0).squeeze()
            tmp = np.where(hv_targets[i, 0, :, :] != 0, 1, 0) * np.where(Temp != 0, 1, 0)
            tmp = 1 - tmp
            hv_targets[i, 0, :, :] = hv_targets[i, 0, :, :] * tmp + Temp

            Temp = np.where(target == inst_id_list[id], tmp_v, 0).squeeze()
            tmp = np.where(hv_targets[i, 1, :, :] != 0, 1, 0) * np.where(Temp != 0, 1, 0)
            tmp = 1 - tmp
            hv_targets[i, 1, :, :] = hv_targets[i, 1, :, :] * tmp + Temp
    hv_targets = torch.from_numpy(hv_targets)
    assert hv_targets.dim() == 4 and hv_targets.shape[1] == 2
    return hv_targets


def get_nc_targets(targets: torch.Tensor) -> torch.Tensor:
    assert targets.dim() == 3
    return targets


def get_inst_centroid(inst_map):
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.int16)
        mask = np.squeeze(mask)
        inst_moment = cv2.moments(mask)
        inst_centroid = ((inst_moment["m10"] / inst_moment["m00"]),  # 横向
                         (inst_moment["m01"] / inst_moment["m00"]))  # 纵向
        inst_centroid_list.append(inst_centroid)
    return inst_centroid_list


def show_hv(img):
    assert img.shape[0] == 2
    assert np.max(img) == 1. and np.min(img) == -1.
    img = img * 122 + 123
    img = img.astype(np.uint8)
    print(img)
    cv2.namedWindow('h')
    cv2.namedWindow('v')
    cv2.imshow('h', img[0, ...])
    cv2.imshow('v', img[1, ...])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_gradient(dh, dv):
    '''

    Parameters
    ----------
    dh  2 dimensional image
    dv  2 dimensional image

    -------

    '''
    assert dh.dim() == 2, dv.dim() == 2
    dh, dv = dh.numpy(), dv.numpy()
    min_dh, max_dh = np.min(dh), np.max(dh)
    min_dv, max_dv = np.min(dv), np.max(dv)
    dh = (dh - min_dh) / (max_dh - min_dh) * 255
    dv = (dv - min_dv) / (max_dv - min_dv) * 255
    dh = dh.astype(np.uint8)
    dv = dv.astype(np.uint8)
    cv2.namedWindow('dh')
    cv2.namedWindow('dv')
    cv2.imshow('dh', dh)
    cv2.imshow('dv', dv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
