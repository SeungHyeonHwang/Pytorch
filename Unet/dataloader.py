# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:40:49 2020

@author: hwang
"""


from os.path import splitext
from os import listdir
import cv2
import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy import io

def basic_dataloader(input_size=128,
                    batch_size=64,
                    num_workers=0,
                    ):
    
    # dir_root = 'C:/Users/USER/Desktop/segmentation/'
    # dir_img = dir_root+ 'data/x/train/'
    # dir_mask = dir_root+ 'data/y/label/'
    # dir_checkpoint = dir_root + 'checkpoints/'
    
    
    DATASET_PATH = 'C:/Users/USER/Desktop/segmentation/'
    train_image_dir = os.path.join(DATASET_PATH, 'data', 'x', 'train') 
    label_path = os.path.join(DATASET_PATH, 'data', 'y2', 'train_label') 

    val_image_dir = os.path.join(DATASET_PATH, 'data', 'x', 'val') 
    val_label_path = os.path.join(DATASET_PATH, 'data', 'y2', 'val_label') 

    train_dataloader = DataLoader(
        BasicDataset(train_image_dir, label_path=label_path, 
                transform=transforms.Compose([
                                              # transforms.ColorJitter(hue=0.1),
                                              transforms.Resize([256, 256]),
                                              # transforms.RandomVerticalFlip(p=0.5),
                                              # transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    
    val_dataloader = DataLoader(
        BasicDataset(val_image_dir, label_path=val_label_path, 
                transform=transforms.Compose([
                                              transforms.Resize([256, 256]),
                                              transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    
    return train_dataloader, val_dataloader


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, label_path=None, transform=None):
        # self.meta_data = meta_data
        self.image_dir = imgs_dir
        self.label_path = label_path
        self.transform = transform
        
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        self.ids_y = [splitext(file)[0] for file in listdir(label_path) if not file.startswith('.')]
        
    def __len__(self):
        # images = glob.glob(self.image_dir+'/*.png')
        return len(self.ids)
    
    # def preprocess(src, islabel=False):
    #     w, h = src.size
        
    #     if islabel :
            
        
    #     return new_src
    
    def __getitem__(self, idx):
        idx_x = self.ids[idx]
        idx_y = self.ids_y[idx]
        
        # print(self.label_path+'/'+ idx_y)
        # print(self.image_dir+'/'+ idx_x)
        # mask_file = glob(self.label_path + idx + '.*')
        img_file = glob(self.image_dir +'/'+ idx_x + '.*')
        mask_file = glob(self.label_path +'/'+ idx_y + '.*')
       
        # print(img_file)
        
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        
        # m = np.array(mask, 'uint8')
        # print('mask : ',m.shape)
        # im = np.array(img, 'uint8')
        # print('img : ',im.shape)     
        
        # print('mask : ',mask_numpy.shape)
        # print('mask max : ', mask_numpy.max())
        # print('img : ', img_numpy.shape)
        # print('img max : ', img_numpy.max())
        # img_pil = Image.fromarray((img_numpy * 255).astype(np.uint8))
        # img_pil = Image.fromarray(img_numpy)
        # mask_pil = Image.fromarray(mask_numpy)
        
        if self.transform:
            new_img = self.transform(img)
            new_mask = self.transform(mask)
        # new_mask = torch.tensor(mask_pil)
        
        mask_numpy = np.array(new_mask, 'float32')
        # mask_numpy = np.where(np.array(mask, 'uint8')>1, 1, 0)
        img_numpy = np.array(new_img, 'float32')
        
        # print(mask_numpy.max(), img_numpy.shape)
        
        return torch.from_numpy(img_numpy), torch.from_numpy(mask_numpy)
        
        
        # img = self.preprocess(img, self.scale)
        # mask = self.preprocess(mask, self.scale)
        
        # return torch.from_numpy(img).type(torch.FloatTensor), torch.from_numpy(mask).type(torch.FloatTensor)
        
    
        # # i=0
        # img_name = os.path.join(self.image_dir + '/'+str(idx).zfill(5)+'.png')
        # # print(img_name)
        # # i+=1
        # mask_name = os.path.join(self.label_path + '/'+str(idx).zfill(5)+'.png')
        # new_img = Image.open(img_name).convert('RGB')
        # mask_img = Image.open(mask_name)
        
        # # img_numpy = np.array(new_img, 'uint8')
        
        
        # # mask_img = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
        # # mask_img = mask_img.astype(np.uint8)
        # # mask_img = ((mask_img - mask_img.min()) * (1/(mask_img.max() - mask_img.min()) * 255)).astype('uint8')
        # # print(mask_img)
        # # mm = Image.fromarray(mask_img)
        # msk_np = np.array(mask_img, 'uint8')
        # # msk_max = msk_np.max()
        
        # # if msk_max > 17 : 
        # mask_img = np.where(msk_np>1, 1, 0)
        
        # # msk_np = ((msk_np - msk_np.min()) * (1/(msk_np.max() - msk_np.min()) * 255)).astype('uint8')
        # # print(msk_np.max())
        
        # # mask_img = np.where(mm>=mm.max()-21, 1, 0)
        # # mask_img = np.where(msk_np>=150, 1, 0)
        
        # # print(mask_img.max())
        # mask_img = Image.fromarray(mask_img)
        
        
        # # random crop 
        # i, j, h, w = transforms.RandomCrop.get_params(mask_img, output_size=(256, 256)) 
        # new_img = transforms.functional.crop(new_img, i, j, h, w) 
        # mask_img = transforms.functional.crop(mask_img, i, j, h, w)

        # # mask_img = mask_img.resize((256, 256))
        
        # trans = transforms.ToTensor()
        
        # msk = trans(mask_img)
        # if self.transform:

        #     new_img = self.transform(new_img)
        #     # print(mask_img)
        #     # mask_img = self.transform(mask_img)
            
            
        # return new_img, msk
        # print(new_img.shape)
        # mask_img = rgb2gray(new_img)
        # mask_img = Image.open(mask_img)
        # mask_img = self.transform(mask_img)
        
        # if self.label_path is not None:
        #     hand_para = self.label_matrix['handPara'][...,idx]
        #     hand_para = torch.tensor(hand_para) # here, we will use only one label among multiple labels.
        #     hand_side = [0,1]
        #     hand_side = torch.tensor(hand_side)
        #     # print(len(new_img))
        #     # print(hand_para.shape)
        #     return new_img, mask_img, hand_para, hand_side
        # else:
        #     return new_img, hand_para, [0,1]
        
        