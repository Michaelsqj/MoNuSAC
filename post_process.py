import os
import torch
from typing import Tuple
import torch.nn.functional as F
import numpy as np
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
                       [-1, 0, 1]], dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=1)
    mv = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=1)

    hl = logits[:, h_ch, :, :].unsqueeze(dim=1).float()
    vl = logits[:, v_ch, :, :].unsqueeze(dim=1).float()

    assert (mh.dim() == 4 and mv.dim() == 4 and hl.dim() == 4 and vl.dim() == 4)

    dh = F.conv2d(hl, mh, stride=1, padding=1)
    dv = F.conv2d(vl, mv, stride=1, padding=1)

    return dh, dv


def get_max_gradient(dh, dv) -> torch.Tensor:
    '''

    Parameters
    ----------
    dh  N*H*W
    dv  N*H*W

    Returns
    -------
    Sm  N*H*W
    '''
    Sm = dh.where(dh > dv, dv)
    assert Sm.shape == dh.shape
    return Sm


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


def post_process(np_logits: torch.Tensor,
                 hv_logits: torch.Tensor,
                 nc_logits: torch.Tensor,
                 thresh_h:float,
                 thresh_k:float)
    assert np_logits.shape[0] == hv_logits.shape[0] == nc_logits.shape[0] == 1
    #暂时只处理单张图片

if __name__ == '__main__':
    dataset = r'E:\MoNUSAC\MyModel\dataset\val\masks\TCGA-55-1594-01Z-00-DX1_004.npy'
    img = np.load(r'E:\MoNUSAC\MyModel\dataset\val\imgs\TCGA-55-1594-01Z-00-DX1_004.npy', allow_pickle=True)
    target = np.load(dataset, allow_pickle=True)
    inst_map = torch.from_numpy(np.squeeze(target[..., 0]))
    class_map = torch.from_numpy(np.squeeze(target[..., 1]))
    hv_target = get_hv_targets(inst_map.unsqueeze(dim=0))
    dh, dv = get_gradient_hv(hv_target)
    Sm = get_max_gradient(dh, dv).squeeze().numpy()

    np_map = np.where(inst_map == 0, 0, 1)
    cv2.namedWindow('np_map')
    cv2.imshow('np_map', np_map.astype(float))
    # cv2.waitKey(0)
    h = 1.5
    gd_map = np.where(Sm > h, 1, 0)

    cv2.namedWindow('gd')
    cv2.imshow('gd', gd_map.astype(float))
    M = np_map - gd_map
    M = np.where(M > 0, M, 0)
    cv2.imshow('M', M.astype(float))
    E = (1 - gd_map) * np_map
    cv2.imshow('E', E.astype(float))
    _, labels = cv2.connectedComponents(E.astype(np.uint8))
    np_map = np_map.astype(np.uint8)
    q = cv2.merge([np_map, np_map, np_map])
    markers = cv2.watershed(q, labels)
    img[markers == -1] = [255, 0, 0]
    cv2.imshow('marker', img)
    cv2.waitKey(0)
