import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-g', '--gpu', default='2')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model4
import utils
import scipy.io as io
import data
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class _HVBranchLoss(nn.Module):
    def __init__(self):
        super(_HVBranchLoss, self).__init__()
        self.mse1 = torch.nn.MSELoss(reduction='mean')
        self.mse2 = torch.nn.MSELoss(reduction='mean')

    def forward(self,
                hv_logits: torch.Tensor,
                hv_targets: torch.Tensor) -> torch.Tensor:
        hv_logits = torch.tanh(hv_logits)
        La = self.mse1(hv_logits, hv_targets.float())

        h_grads, v_grads = utils.get_gradient_hv(hv_logits, h_ch=0, v_ch=1)
        h_grads_tar, v_grads_tar = utils.get_gradient_hv(hv_targets, h_ch=0, v_ch=1)

        h_grads, v_grads, h_grads_tar, v_grads_tar = h_grads.cuda(), v_grads.cuda(), h_grads_tar.cuda(), v_grads_tar.cuda()
        Lb = self.mse2(h_grads.squeeze(), h_grads_tar.squeeze().float()) + F.mse_loss(v_grads.squeeze(),
                                                                                      v_grads_tar.squeeze().float())
        print(f'La{La.item()},  Lb{Lb.item()}')
        loss = 10 * La + 0*Lb
        return loss


class HoverLoss(nn.Module):
    def __init__(self):
        super(HoverLoss, self).__init__()
        self.hv_loss = _HVBranchLoss()

    def forward(self, hv_logits, hv_targets) -> torch.Tensor:
        loss = self.hv_loss(hv_logits, hv_targets)
        return loss


class predict(nn.Module):
    def __init__(self, datafolder, model_save_path):
        super(predict, self).__init__()
        self.batch_size = 1
        self.dataloader = data.provider(data_folder=datafolder, phase='train', input_shape=(256, 256), batch_size=1)
        self.model = model4.HoverNet()
        self.model.load_state_dict(torch.load(model_save_path))
        # self.model.eval()
        self.hoverloss = HoverLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def start(self):
        _, (image, target) = enumerate(self.dataloader).__next__()
        image = image.to(self.device).float()
        hv_logits = self.model(image)

        hv_targets = utils.get_hv_targets(target[:, 0, :, :])
        hv_targets = hv_targets.to(self.device)
        loss = self.hoverloss(hv_logits, hv_targets)

        h_grads, v_grads = utils.get_gradient_hv(hv_logits, h_ch=0, v_ch=1)
        h_grads_tar, v_grads_tar = utils.get_gradient_hv(hv_targets, h_ch=0, v_ch=1)

        hv_grads = torch.cat([h_grads, v_grads], dim=1)
        hv_grads_tar = torch.cat([h_grads_tar, v_grads_tar], dim=1)

        print(f'loss:{loss.item()}')
        with torch.no_grad():
            np.save('/home/jqshen/MyCode/MyModel/prediction/hv_logits.npy', hv_logits.cpu().numpy())
            np.save('/home/jqshen/MyCode/MyModel/prediction/hv_target.npy', hv_targets.cpu().numpy())
            np.save('/home/jqshen/MyCode/MyModel/prediction/hv_grads.npy', hv_grads.cpu().numpy())
            np.save('/home/jqshen/MyCode/MyModel/prediction/hv_grads_tar.npy', hv_grads_tar.cpu().numpy())
            np.save('/home/jqshen/MyCode/MyModel/prediction/hv_image.npy', image.cpu().numpy())


if __name__ == '__main__':
    data_dir = '/home/jqshen/MyCode/MyModel/dataset'
    model_save_path = '/home/jqshen/MyCode/MyModel/checkpoints/hover4.tar'
    tester = predict(datafolder=data_dir, model_save_path=model_save_path)
    tester.start()
