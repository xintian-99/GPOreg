import math
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from torch.nn import functional as F

class DiceLoss(nn.Module):
    """Dice loss"""

    def __init__(self, num_class=14, is_square=False):
        super().__init__()
        self.num_class = num_class
        self.is_square = is_square

    def forward(self, y_pred, y_true):
        '''
        Assuming y_pred has been one-hot encoded: [bs, num_class, h, w, d]
        '''
        y_true = nn.functional.one_hot(y_true.long(), num_classes=self.num_class)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()

        if y_pred.shape[2] != y_true.shape[2] or y_pred.shape[3] != y_true.shape[3] or y_pred.shape[4] != y_true.shape[4]:
            y_pred = nn.functional.interpolate(y_pred, size=(y_true.shape[2], y_true.shape[3], y_true.shape[4]), mode='trilinear', align_corners=True)

        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3, 4])
        if self.is_square:
            union = torch.pow(y_pred, 2).sum(dim=[2, 3, 4]) + torch.pow(y_true, 2).sum(dim=[2, 3, 4])
        else:
            union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))

        return dsc

class DiceLoss2D(nn.Module):
    """Dice loss"""

    def __init__(self, num_class=14, is_square=False):
        super().__init__()
        self.num_class = num_class
        self.is_square = is_square

    def forward(self, y_pred, y_true):
        '''
        Assuming y_pred has been one-hot encoded: [bs, num_class, h, w, d]
        '''
        y_true = nn.functional.one_hot(y_true.long(), num_classes=self.num_class)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 3, 1, 2).contiguous()

        if y_pred.shape[2] != y_true.shape[2] or y_pred.shape[3] != y_true.shape[3]:
            y_pred = nn.functional.interpolate(y_pred, size=(y_true.shape[2], y_true.shape[3]), mode='bilinear', align_corners=True)

        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3])

        if self.is_square:
            union = torch.pow(y_pred, 2).sum(dim=[2, 3]) + torch.pow(y_true, 2).sum(dim=[2, 3])
        else:
            union = y_pred.sum(dim=[2, 3]) + y_true.sum(dim=[2, 3])
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))

        return dsc

class BinaryDiceLoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):

        y_pred = y_pred.float()
        y_true = y_true.float()

        intersection = y_pred * y_true
        intersection = intersection.sum(dim=(2,3,4))
        union = y_pred.sum(dim=(2,3,4)) + y_true.sum(dim=(2,3,4))
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))

        return dsc

class Grad4d(nn.Module):

    def __init__(self, penalty='l1'):
        super(Grad4d, self).__init__()

        self.penalty = penalty

    def forward(self, y_pred, y_true=None):

        dx = ((y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])**2).mean()
        dy = ((y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])**2).mean()
        dz = ((y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])**2).mean()
        dd = ((y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])**2).mean()

        grad = (dx + dy + dz + dd) / 4.0

        return grad

class Grad3d(nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self):
        super(Grad3d, self).__init__()

    def forward(self, y_pred):

        dy = ((y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])**2).mean()
        dx = ((y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])**2).mean()
        dz = ((y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])**2).mean()
        grad = (dy + dx + dz) / 3.0

        return grad

class Grad2d(nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l1'):
        super(Grad2d, self).__init__()

        self.penalty = penalty

    def forward(self, y_pred, y_true=None):

        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        return grad

class NccLoss(nn.Module):

    def __init__(self, win=None):
        super(NccLoss, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred, mask=None):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_true.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        if mask is not None:
            mask = mask.float()
            cc = cc * mask

        return 1.-torch.mean(cc)


class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + e))
            * (tensor.detach() - tensor.mean().detach())
        )

class GccLoss(_Loss):
    def __init__(self, use_mask: bool = False):
        super().__init__()
        self.forward = self.metric

    def ncc(self, x1, x2, e=1e-10):
        assert x1.shape == x2.shape, "Inputs are not of similar shape"
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        stablestd = StableStd.apply
        std = stablestd(x1) * stablestd(x2)
        ncc = cc / (std + e)
        return ncc

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        return 1-self.ncc(fixed, warped).mean()

class GmiLoss(_Loss):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, eps=1e-7):

        nx = x.shape[0]
        ny = y.shape[0]
        n_voxels = x.shape[1]

        H_x = 0
        for i in range(nx):
            x_i_sum = x[i].sum()
            p = x_i_sum / n_voxels
            mi_i = -p*torch.log2(torch.clamp(p, min=eps, max=None))
            H_x += mi_i

        H_y = 0
        for i in range(ny):
            y_i_sum = y[i].sum()
            if y_i_sum < 2:
                continue
            p = y_i_sum / n_voxels
            mi_i = -p*torch.log2(torch.clamp(p, min=eps, max=None))
            H_y += mi_i

        H_xy = 0
        for i in range(nx):
            for j in range(ny):
                xy_ij_sum = (x[i]*y[j]).sum()
                if xy_ij_sum < 2:
                    continue
                p = xy_ij_sum / n_voxels
                mi_ij = -p*torch.log2(torch.clamp(p, min=eps, max=None))
                H_xy += mi_ij

        cmif = H_x + H_y - H_xy

        return cmif