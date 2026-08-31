import re
import os
import glob
import scipy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from collections import deque, OrderedDict
from utils.surface_distance import compute_robust_hausdorff, compute_surface_distances

def get_downsampled_images(img, n_downs=4, mode='trilinear', n_cs=1):

    blur = GaussianBlur3D(n_cs, sigma=1).to(img.device)
    out_imgs = [img]
    for _ in range(n_downs):
        img = blur(img)
        img = F.interpolate(img, scale_factor=0.5, mode=mode, align_corners=True)
        out_imgs.append(img)

    return out_imgs

def blur_axial(img, blur):
    xs = torch.chunk(img, 16, dim=-1)
    outs = []
    for x in xs:
        outs.append(blur(x.squeeze(-1)).unsqueeze(-1))
    return torch.cat(outs, dim=-1)


def get_downsampled_images_2dAxial(img, n_downs=4, scale=(.5,.5,1), mode='bilinear', n_cs=1):

    blur = GaussianBlur2D(n_cs, sigma=1).to(img.device)
    out_imgs = [img]
    for _ in range(n_downs):
        img = blur_axial(img, blur)
        img = F.interpolate(img, scale_factor=scale,
                            mode=mode, align_corners=True)
        out_imgs.append(img)

    return out_imgs

def get_downsampled_images_2d(img, n_downs=4, mode='bilinear', n_cs=1):

    if n_cs > 0:
        blur = GaussianBlur2D(n_cs, sigma=1).to(img.device)
    out_imgs = [img]
    for _ in range(n_downs):
        if n_cs > 0:
            img = blur(img)
        if mode == 'nearest':
            img = F.interpolate(img, scale_factor=0.5, mode=mode)
        else:
            img = F.interpolate(img, scale_factor=0.5, mode=mode, align_corners=True)
        out_imgs.append(img)

    return out_imgs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, is_grid_out=False, mode=None, align_corners=True):

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if mode is None:
            out = F.grid_sample(src, new_locs, align_corners=align_corners, mode=self.mode)
        else:
            out = F.grid_sample(src, new_locs, align_corners=align_corners, mode=mode)

        if is_grid_out:
            return out, new_locs
        return out

class registerSTModel(nn.Module):

    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(registerSTModel, self).__init__()

        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, img, flow, is_grid_out=False, align_corners=True):

        out = self.spatial_trans(img,flow,is_grid_out=is_grid_out,align_corners=align_corners)

        return out

class VecInt(nn.Module):

    def __init__(self, inshape, nsteps):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps

        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):

        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)

        return vec

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)
    x = param_group['lr']
    return x

def dice_eval(y_pred, y_true, num_cls, exclude_background=True, output_individual=False):

    y_pred = nn.functional.one_hot(y_pred, num_classes=num_cls)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_cls)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)

    dscs = []
    if output_individual:
        dscs = [torch.mean(dsc[:,x:x+1]) for x in range(1,num_cls)]

    if exclude_background:
        out = [torch.mean(torch.mean(dsc[:,1:], dim=1))] + dscs
    else:
        out = [torch.mean(torch.mean(dsc, dim=1))] + dscs

    if len(out) == 1:
        return out[0]
    return tuple(out)

def dice_eval_2D(y_pred, y_true, num_cls, exclude_background=True, output_individual=False):

    y_pred = nn.functional.one_hot(y_pred, num_classes=num_cls)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 3, 1, 2).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_cls)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 3, 1, 2).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3])
    union = y_pred.sum(dim=[2, 3]) + y_true.sum(dim=[2, 3])
    dsc = (2.*intersection) / (union + 1e-5)

    dscs = []
    if output_individual:
        dscs = [torch.mean(dsc[:,x:x+1]) for x in range(1,num_cls)]

    if exclude_background:
        out = [torch.mean(torch.mean(dsc[:,1:], dim=1))] + dscs
    else:
        out = [torch.mean(torch.mean(dsc, dim=1))] + dscs

    if len(out) == 1:
        return out[0]
    return tuple(out)


def convert_pytorch_grid2scipy(grid):

    _, H, W, D = grid.shape
    grid_x = (grid[0, ...] + 1) * (D -1)/2
    grid_y = (grid[1, ...] + 1) * (W -1)/2
    grid_z = (grid[2, ...] + 1) * (H -1)/2

    grid = np.stack([grid_z, grid_y, grid_x])

    identity_grid = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    grid = grid - identity_grid

    return grid

# def dice_missing_eval(y_pred, y_true, num_cls, exclude_background=True, output_individual=False):

#     for i in range(1, num_cls):
#         if torch.sum(y_true == i) == 0:
#             y_true = y_true + (y_pred == i).float()

def dice_binary(pred, truth, k = 1):
    truth[truth!=k]=0
    pred[pred!=k]=0
    truth=truth/k
    pred=pred/k
    intersection = np.sum(pred[truth==1.0]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(truth)+1e-7)

    return dice

def compute_tre(x, y, spacing):
    return np.linalg.norm((x - y) * spacing, axis=1)


class modelSaver():

    def __init__(self, save_path, save_freq, n_checkpoints = 10):

        self.save_path = save_path
        self.save_freq = save_freq
        self.best_score = -1e6
        self.best_loss = 1e6
        self.n_checkpoints = n_checkpoints
        self.epoch_fifos = deque([])
        self.score_fifos = deque([])
        self.loss_fifos = deque([])

        self.initModelFifos()

    def initModelFifos(self):

        epoch_epochs = []
        score_epochs = []
        loss_epochs  = []

        file_list = glob.glob(os.path.join(self.save_path, '*epoch*.pth'))
        if file_list:
            for file_ in file_list:
                file_name = "net_epoch_(.*)_score_.*.pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    epoch_epochs.append(int(result[0]))

                file_name = "best_score_.*_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    score_epochs.append(int(result[0]))

                file_name = "best_loss_.*_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    loss_epochs.append(int(result[0]))

        score_epochs.sort()
        epoch_epochs.sort()
        loss_epochs.sort()

        if file_list:
            for file_ in file_list:
                for epoch_epoch in epoch_epochs:
                    file_name = "net_epoch_" + str(epoch_epoch) + "_score_.*.pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.epoch_fifos.append(result[0])

                for score_epoch in score_epochs:
                    file_name = "best_score_.*_net_epoch_" + str(score_epoch) +".pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.score_fifos.append(result[0])

                for loss_epoch in loss_epochs:
                    file_name = "best_loss_.*_net_epoch_" + str(loss_epoch) +".pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.loss_fifos.append(result[0])

        print("----->>>> BEFORE: epoch_fifos length: %d, score_fifos_length: %d, loss_fifos_length: %d" % (len(self.epoch_fifos), len(self.score_fifos), len(self.loss_fifos)))

        self.updateFIFOs()

        print("----->>>> AFTER: epoch_fifos length: %d, score_fifos_length: %d, loss_fifos_length: %d" % (len(self.epoch_fifos), len(self.score_fifos), len(self.loss_fifos)))

    def saveModel(self, model, epoch, avg_score, loss=None):

        torch.save(model.state_dict(), os.path.join(self.save_path, 'net_latest.pth'))

        if epoch % self.save_freq == 0:

            file_name = ('net_epoch_%d_score_%.4f.pth' % (epoch, avg_score))
            self.epoch_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        if avg_score >= self.best_score:

            self.best_score = avg_score
            file_name = ('best_score_%.4f_net_epoch_%d.pth' % (avg_score, epoch))
            self.score_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        if loss is not None and loss <= self.best_loss:

            self.best_loss = loss
            file_name = ('best_loss_%.4f_net_epoch_%d.pth' % (loss, epoch))
            self.loss_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        self.updateFIFOs()

    def updateFIFOs(self):

        while(len(self.epoch_fifos) > self.n_checkpoints):
            file_name = self.epoch_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        while(len(self.score_fifos) > self.n_checkpoints):
            file_name = self.score_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        while(len(self.loss_fifos) > self.n_checkpoints):
            file_name = self.loss_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

def convert_state_dict(state_dict, is_multi = False):

    new_state_dict = OrderedDict()

    if is_multi:
        if next(iter(state_dict)).startswith("module."):
            return state_dict  # abort if dict is a DataParallel model_state

        for k, v in state_dict.items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
    else:

        if not next(iter(state_dict)).startswith("module."):
            return state_dict  # abort if dict is not a DataParallel model_state

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

    return new_state_dict

def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])

    return jacdet

def jacobian_determinant_2d(disp):
    # Assuming disp has shape [2, H, W], representing displacement in two dimensions
    H, W = disp.shape[1], disp.shape[2]

    # Define gradients for x and y directions
    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3)

    # Compute gradients of displacement components
    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :], grady, mode='constant', cval=0.0)], axis=1)

    # Stack gradients to form the Jacobian matrix for each point
    grad_disp = np.concatenate([gradx_disp, grady_disp], 0)
    # Add identity matrix since the displacement is relative to an identity grid
    jacobian = grad_disp + np.eye(2).reshape(2, 2, 1, 1)

    # Crop the edges to reduce edge effects
    # Note: Adjust this line if you need a different cropping strategy
    jacobian_cropped = jacobian[:, :, 2:-2, 2:-2]

    # Calculate determinant of the 2x2 Jacobian at each point
    jacdet = jacobian_cropped[0, 0, :, :] * jacobian_cropped[1, 1, :, :] - jacobian_cropped[0, 1, :, :] * jacobian_cropped[1, 0, :, :]

    return jacdet

def compute_HD95(moving, fixed, moving_warped,num_classes=14,spacing=np.ones(3)):

    hd95 = 0
    count = 0
    for i in range(1, num_classes):
        if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
            continue
        if ((moving_warped==i).sum()==0):
            hd95 += compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving==i), spacing), 95.)
        else:
            hd95 += compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving_warped==i), spacing), 95.)
        count += 1
    hd95 /= count

    return hd95

def computeJacDetVal(jac_det, img_size):

    jac_det_val = np.sum(jac_det <= 0) / np.prod(img_size)

    return jac_det_val

def computeSDLogJ(jac_det, rho=3):

    log_jac_det = np.log(np.abs((jac_det+rho).clip(1e-8, 1e8)))
    std_dev_jac = np.std(log_jac_det)

    return std_dev_jac

class GaussianBlur3D(nn.Module):
    def __init__(self, channels, sigma=1, kernel_size=0):
        super(GaussianBlur3D, self).__init__()
        self.channels = channels
        if kernel_size == 0:
            kernel_size = int(2.0 * sigma * 2 + 1)

        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size**2).view(kernel_size, kernel_size, kernel_size)
        y_grid = x_grid.transpose(0, 1)
        z_grid = x_grid.transpose(0, 2)
        xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.

        gaussian_kernel = (1. / (2. * np.pi * variance) ** 1.5) * \
                          torch.exp(
                              -torch.sum((xyz_grid - mean) ** 2., dim=-1) /
                              (2 * variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

        self.register_buffer('gaussian_kernel', gaussian_kernel)
        self.padding = kernel_size // 2

    def forward(self, x):
        blurred = F.conv3d(x, self.gaussian_kernel, padding=self.padding, groups=self.channels)
        return blurred

class GaussianBlur2D(nn.Module):
    def __init__(self, channels, sigma=1, kernel_size=0):
        super(GaussianBlur2D, self).__init__()
        self.channels = channels

        if kernel_size == 0:
            kernel_size = int(2.0 * sigma * 2 + 1)

        # Create a 2D Gaussian kernel
        coord = torch.arange(kernel_size)
        grid = coord.repeat(kernel_size).view(kernel_size, kernel_size)
        xy_grid = torch.stack([grid, grid.t()], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.

        gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) / \
                              (2 * variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.register_buffer('gaussian_kernel', gaussian_kernel)
        self.padding = kernel_size // 2

    def forward(self, x):
        blurred = F.conv2d(x, self.gaussian_kernel, padding=self.padding, groups=self.channels)
        return blurred

class AnisotropicGaussianBlur3D(nn.Module):
    def __init__(self, channels, sigma=(1, 1, 1), kernel_size=0):
        super(AnisotropicGaussianBlur3D, self).__init__()
        self.channels = channels
        sigma_d, sigma_h, sigma_w = sigma

        if kernel_size == 0:
            kernel_size_d = int(2.0 * sigma_d * 2 + 1)
            kernel_size_h = int(2.0 * sigma_h * 2 + 1)
            kernel_size_w = int(2.0 * sigma_w * 2 + 1)
        else:
            kernel_size_d = kernel_size_h = kernel_size_w = kernel_size

        # Create a 3D Gaussian kernel for each dimension
        d_coord = torch.arange(kernel_size_d)
        h_coord = torch.arange(kernel_size_h)
        w_coord = torch.arange(kernel_size_w)

        d_grid = d_coord.repeat(kernel_size_h, kernel_size_w, 1).permute(2, 0, 1)
        h_grid = h_coord.repeat(kernel_size_d, kernel_size_w, 1).permute(1, 0, 2)
        w_grid = w_coord.repeat(kernel_size_d, kernel_size_h, 1).permute(1, 2, 0)

        mean_d = (kernel_size_d - 1) / 2.
        mean_h = (kernel_size_h - 1) / 2.
        mean_w = (kernel_size_w - 1) / 2.

        variance_d = sigma_d ** 2.
        variance_h = sigma_h ** 2.
        variance_w = sigma_w ** 2.

        # Calculate the Gaussian kernel
        gaussian_kernel = (1. / ((2. * np.pi) ** 1.5 * sigma_d * sigma_h * sigma_w)) * \
                          torch.exp(-(((d_grid - mean_d) ** 2.) / (2 * variance_d) +
                                      ((h_grid - mean_h) ** 2.) / (2 * variance_h) +
                                      ((w_grid - mean_w) ** 2.) / (2 * variance_w)))

        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 3d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size_d, kernel_size_h, kernel_size_w)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

        self.register_buffer('gaussian_kernel', gaussian_kernel)
        self.padding_d = kernel_size_d // 2
        self.padding_h = kernel_size_h // 2
        self.padding_w = kernel_size_w // 2

    def forward(self, x):
        x = F.pad(x, (self.padding_w, self.padding_w, self.padding_h, self.padding_h, self.padding_d, self.padding_d), mode='replicate')
        blurred = F.conv3d(x, self.gaussian_kernel, groups=self.channels)
        return blurred