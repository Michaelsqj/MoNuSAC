import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
import scipy.io as io


def show():
    dataset = r'E:\MoNUSAC\MyModel\prediction'
    for file in os.listdir(dataset):
        cv2.imshow(file, np.load(os.path.join(dataset, file), allow_pickle=True))
    cv2.waitKey(0)


def plot():
    dataset = r'E:\MoNUSAC\MyModel\hover.npy'
    store = np.load(dataset, allow_pickle=True).item()
    for metric in store.keys():
        plt.figure()
        plt.plot(range(len(store[metric]["val"])),
                 store[metric]["val"],
                 label=f'val {metric}')
        # plt.plot(range(len(scores["val"])),
        #          scores["val"],
        #          label=f'val {name}')
        plt.title(f'{metric} plot')
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric}')
        plt.legend()
        plt.savefig(f'E:\MoNUSAC\MyModel\{metric}_val.jpg')


def hv():
    # nc_logits=np.load(r'E:\MoNUSAC\MyModel\prediction\nc_logits.npy',allow_pickle=True)
    hv_logits = np.load(r'E:\MoNUSAC\MyModel\prediction\hv_logits.npy', allow_pickle=True)
    # nc_target=np.load(r'E:\MoNUSAC\MyModel\prediction\nc_target.npy',allow_pickle=True)
    hv_targets = np.load(r'E:\MoNUSAC\MyModel\prediction\hv_target.npy', allow_pickle=True)
    hv_grads = np.load(r'E:\MoNUSAC\MyModel\prediction\hv_grads.npy', allow_pickle=True)
    hv_grads_tar = np.load(r'E:\MoNUSAC\MyModel\prediction\hv_grads_tar.npy', allow_pickle=True)
    # nc_logits=np.squeeze(np.argmax(nc_logits,axis=1)).astype(np.uint8)
    # nc_logits=nc_logits*60
    # np_logits=np.squeeze(np.argmax(np_logits,axis=1)).astype(np.uint8)
    # np_logits=np_logits*255
    # np_target=np.squeeze(np_target).astype(np.uint8)
    # np_target=np_target*255
    # cv2.imshow('nptarget',np_target)
    # cv2.imshow('np',np_logits)
    # cv2.imshow('nc',nc_logits)
    # nc_target=np.squeeze(nc_target).astype(np.uint8)
    # nc_target=nc_target*60
    # cv2.imshow('nctarget',nc_target)
    # cv2.waitKey(0)
    hv_logits = np.squeeze(hv_logits)
    h, v = np.squeeze(hv_logits[0, :, :]).astype(np.float), np.squeeze(hv_logits[1, :, :]).astype(np.float)

    h, v = (h - np.min(h)) / (np.max(h) - np.min(h)) * 255, (v - np.min(v)) / (np.max(v) - np.min(v)) * 255
    cv2.imshow('h', h.astype(np.uint8))
    cv2.imshow('v', v.astype(np.uint8))

    hv_targets = np.squeeze(hv_targets)
    ht, vt = np.squeeze(hv_targets[0, :, :]).astype(np.float), np.squeeze(hv_targets[1, :, :]).astype(np.float)
    print(np.max(ht), np.min(ht))
    ht, vt = (ht - np.min(ht)) / (np.max(ht) - np.min(ht)) * 255, (vt - np.min(vt)) / (np.max(vt) - np.min(vt)) * 255
    cv2.imshow('ht', ht.astype(np.uint8))
    cv2.imshow('vt', vt.astype(np.uint8))

    hv_grads = np.squeeze(hv_grads)
    h_grads, v_grads = np.squeeze(hv_grads[0, :, :]).astype(np.float), np.squeeze(hv_grads[1, :, :]).astype(np.float)
    h_grads, v_grads = (h_grads - np.min(h_grads)) / (np.max(h_grads) - np.min(h_grads)) * 255, (
            v_grads - np.min(v_grads)) / (np.max(v_grads) - np.min(v_grads)) * 255
    cv2.imshow('h_grads', h_grads.astype(np.uint8))
    cv2.imshow('v_grads', v_grads.astype(np.uint8))

    hv_grads_tar = np.squeeze(hv_grads_tar)
    h_grads_tar, v_grads_tar = np.squeeze(hv_grads_tar[0, :, :]).astype(np.float), np.squeeze(
        hv_grads_tar[1, :, :]).astype(np.float)
    h_grads_tar, v_grads_tar = (h_grads_tar - np.min(h_grads_tar)) / (
            np.max(h_grads_tar) - np.min(h_grads_tar)) * 255, (v_grads_tar - np.min(v_grads_tar)) / (
                                       np.max(v_grads_tar) - np.min(v_grads_tar)) * 255
    cv2.imshow('h_grads_tar', h_grads_tar.astype(np.uint8))
    cv2.imshow('v_grads_tar', v_grads_tar.astype(np.uint8))

    # image = np.squeeze(np.load(r'E:\MoNUSAC\MyModel\prediction\hv_image.npy', allow_pickle=True))
    # image=image.transpose((1,2,0))
    # cv2.imshow('image', image)
    cv2.waitKey(0)


def npnc():
    nc_logits = np.squeeze(np.load(r'E:\MoNUSAC\MyModel\prediction\nc_logits.npy', allow_pickle=True))
    nc_target = np.squeeze(np.load(r'E:\MoNUSAC\MyModel\prediction\nc_target.npy', allow_pickle=True)) * 60
    np_target = np.squeeze(np.load(r'E:\MoNUSAC\MyModel\prediction\np_target.npy', allow_pickle=True)) * 255
    np_logits = np.squeeze(np.load(r'E:\MoNUSAC\MyModel\prediction\np_logits.npy', allow_pickle=True))
    image = np.squeeze(np.load(r'E:\MoNUSAC\MyModel\prediction\image.npy', allow_pickle=True))
    print(image.shape)
    image = image.transpose((1, 2, 0))
    nc_logits = np.argmax(nc_logits, axis=0) * 60
    np_logits = np.argmax(np_logits, axis=0) * 255
    cv2.imshow('np_target', np_target.astype(np.uint8))
    cv2.imshow('np_logits', np_logits.astype(np.uint8))
    cv2.imshow('nc_target', nc_target.astype(np.uint8))
    cv2.imshow('nc_logits', nc_logits.astype(np.uint8))
    cv2.imshow('image', image)
    # cv2.waitKey(0)


def show_(winname, img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img = img.astype(np.uint8)
    cv2.imshow(winname, img)


def post_process(q=-5):
    nc_logits = np.squeeze(np.squeeze(np.load(r'E:\MoNUSAC\MyModel\prediction\nc_logits.npy', allow_pickle=True)))
    nc_target = np.squeeze(np.squeeze(np.load(r'E:\MoNUSAC\MyModel\prediction\nc_target.npy', allow_pickle=True)))
    np_logits = np.squeeze(np.squeeze(np.load(r'E:\MoNUSAC\MyModel\prediction\np_logits.npy', allow_pickle=True)))
    hv_logits = np.squeeze(np.load(r'E:\MoNUSAC\MyModel\prediction\hv_logits.npy', allow_pickle=True))
    image = np.squeeze(np.load(r'E:\MoNUSAC\MyModel\prediction\image.npy', allow_pickle=True))
    image = image.transpose((1, 2, 0))
    image2 = image.copy()
    image2 += 60 * cv2.merge(
        [np.where(nc_target == 1, 1, 0), np.where(nc_target == 2, 2, 0) + np.where(nc_target == 4, 4, 0),
         np.where(nc_target == 3, 3, 0) + np.where(nc_target == 4, 4, 0)])
    cv2.imshow('target', image2)
    # cv2.imshow('h',np.squeeze(hv_logits[0,...]).astype(np.float))
    h, v = cv2.Sobel(np.squeeze(hv_logits[0, :, :]), ddepth=-1, dx=1, dy=0, ksize=3), cv2.Sobel(
        np.squeeze(hv_logits[1, :, :]),
        ddepth=-1,
        dx=0, dy=1, ksize=3)
    # show_('h',h)
    # show_('v',v)
    # cv2.imshow('H', h.astype(np.float))
    # cv2.imshow('V', v.astype(np.float))
    Sm = np.where(h < v, h, v)
    t = np.argmax(np_logits, axis=0)
    # t=np_logits
    # cv2.imshow('sm', Sm.astype(np.float))
    # show_('Sm',Sm)

    tmp = t - np.where(Sm < q, 1, 0)
    tmp = np.where(tmp > 0, 1, 0)

    # cv2.imshow('tmp',tmp.astype(np.uint8)*255)
    sure_bg = 1 - t
    sure_fg = tmp
    unkown = 1 - sure_bg - sure_fg
    _, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
    markers = markers + 1
    markers[sure_bg == 1] = 0
    # cv2.imshow('mark',markers.astype(np.uint8))
    img = cv2.distanceTransform(t.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_3)
    img = (np.max(img) - img) / (np.max(img) - np.min(img)) * 255
    img = img.astype(np.uint8)
    img = cv2.merge([img, img, img])
    # cv2.imshow('img',img)
    markers = cv2.watershed(img, markers)
    m = np.zeros((256, 256))
    nc_logits = np.argmax(nc_logits, axis=0)
    for i in np.unique(markers).tolist():
        if i > 2:
            temp = markers.copy()
            temp = np.where(temp == i, 1, 0)
            temp = temp * nc_logits
            maxs = 0
            maxc = 0
            for j in np.unique(temp).tolist():
                if j > 0:
                    temp2 = np.where(temp == j, 1, 0)
                    s = np.sum(temp2)
                    if s > maxs:
                        maxs = s
                        maxc = j
            m += np.where(markers == i, maxc, 0)

    image += 60 * cv2.merge(
        [np.where(m == 1, 1, 0), np.where(m == 2, 2, 0) + np.where(m == 4, 4, 0),
         np.where(m == 3, 3, 0) + np.where(m == 4, 4, 0)])
    image[markers == -1] = [0, 0, 255]
    cv2.imshow('img', image)
    cv2.waitKey(0)


# npnc()
# hv()
post_process()
# print(np.ones((3)).tolist())
