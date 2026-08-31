import os
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def rgb2gray(rgb):
    """ Convert RGB image to gray image (extract green channel) """
    r, g, b = rgb.split()
    return g

def clahe_equalized(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    images_equalized = clahe.apply(np.array(images, dtype=np.uint8))

    return images_equalized

def datasets_normalized(images):
    images_std = np.std(images)
    images_mean = np.mean(images)
    images_normalized = (images - images_mean) / (images_std + 1e-6)
    minv = np.min(images_normalized)
    maxv = np.max(images_normalized)
    images_normalized = ((images_normalized - minv) / (maxv - minv + 1e-6)) * 255

    return images_normalized

def adjust_gamma(images, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    new_images = cv2.LUT(np.array(images, dtype=np.uint8), table)

    return new_images

def pre_processing(data):
    """ Enhance retinal images """
    data = rgb2gray(data)
    data = np.array(data)
    mask = (data > 0).astype(np.float32)
    train_imgs = datasets_normalized(data)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)

    train_imgs = train_imgs / 255.

    return train_imgs.astype(np.float32) * mask

class firereg_loader(Dataset): # firereg_loader

    list_a = ['A'+str(idx).zfill(2) for idx in range(1, 14+1)]
    list_p = ['P'+str(idx).zfill(2) for idx in range(1, 49+1)]
    list_s = ['S'+str(idx).zfill(2) for idx in range(1, 71+1)]
    lists = {'A': list_a, 'P': list_p, 'S': list_s}

    def __init__(self,
            root_dir = './../../../data/firereg/',
            split = 'A', # A list, P list or S list or test
            geo_num = 1000,
            enable_geo_init: bool = True,
        ):
        self.root_dir = root_dir
        self.split = split
        self.geo_num = geo_num
        self.enable_geo_init = bool(enable_geo_init)
        train_r, val_r, test_r = 0.7, 0.1, 0.2
        
        valid = {'A','P','S','train','val','test'}
        if split not in valid:
            raise ValueError(f"split must be one of {sorted(valid)}, got {split}")
        
        len_a, len_p, len_s = len(self.list_a), len(self.list_p), len(self.list_s)
        if self.split == 'train':
            a_list = self.list_a[:int(len_a*train_r)]
            p_list = self.list_p[:int(len_p*train_r)]
            s_list = self.list_s[:int(len_s*train_r)]
        elif self.split == 'val':
            a_list = self.list_a[int(len_a*train_r):int(len_a*(train_r+val_r))]
            p_list = self.list_p[int(len_p*train_r):int(len_p*(train_r+val_r))]
            s_list = self.list_s[int(len_s*train_r):int(len_s*(train_r+val_r))]
        elif self.split == 'test':
            a_list = self.list_a[int(len_a*(train_r+val_r)):]
            p_list = self.list_p[int(len_p*(train_r+val_r)):]
            s_list = self.list_s[int(len_s*(train_r+val_r)):]
        
        if self.split in ['A', 'P', 'S']:
            self.total_list = self.lists[self.split]
        else:
            self.total_list = a_list + p_list + s_list        
        if 'P37' in self.total_list:
            self.total_list.remove('P37')

        print('Total number of {} samples: {}'.format(self.split, len(self.total_list)))

        self.img_folder = 'img'
        self.kps_folder = 'gt_kps'
        self.geo_folder = f'geo_kps_{self.geo_num}'
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):

        sub_idx_str = self.total_list[idx]

        img_x_fp = os.path.join(self.root_dir,self.img_folder,sub_idx_str+'_1.jpg')
        img_y_fp = os.path.join(self.root_dir,self.img_folder,sub_idx_str+'_2.jpg')

        img_x = Image.open(img_x_fp).convert('RGB')
        img_y = Image.open(img_y_fp).convert('RGB')

        img_x = self.to_tensor(pre_processing(img_x))
        img_y = self.to_tensor(pre_processing(img_y))

        kps1, kps2 = self.load_keypoints(sub_idx_str)
        kps1 = torch.tensor(kps1, dtype=torch.float32)
        kps2 = torch.tensor(kps2, dtype=torch.float32)

        if self.enable_geo_init:
            geo1, geo2 = self.load_geopoints(sub_idx_str)
            geo1 = torch.tensor(geo1, dtype=torch.float32)
            geo2 = torch.tensor(geo2, dtype=torch.float32)
        else:
            geo1 = torch.empty((0, 2), dtype=torch.float32)
            geo2 = torch.empty((0, 2), dtype=torch.float32)

        return img_x, img_y, kps1, kps2, geo1, geo2, sub_idx_str

    def load_keypoints(self, sub_idx_str):

        kps_path = os.path.join(self.root_dir,self.kps_folder,'control_points_'+sub_idx_str+'_1_2.txt')

        kps = pd.read_csv(kps_path, header=None, sep='\s+')
        kps1 = kps[[0, 1]].values
        kps2 = kps[[2, 3]].values

        return kps1, kps2

    def load_geopoints(self, sub_idx_str):

        kps_path = os.path.join(self.root_dir,self.geo_folder,sub_idx_str+'_1_2.csv')
        
        if not os.path.exists(kps_path):
            raise FileNotFoundError(
                f"Geo init enabled but missing file: {kps_path}. "
                f"Either generate geo_kps_{self.geo_num}/*.csv or run with --enable_geo_init 0."
            )
            
        kps = pd.read_csv(kps_path, header=None)
        kps1 = kps[[0, 1]].values
        kps2 = kps[[2, 3]].values

        return kps1, kps2