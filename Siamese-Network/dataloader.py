

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# # %matplotlib inline
import random


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

# train pair
def train_dataloader(dataset, input_size=64, batch_size=16, num_workers=0):
    

    train_dataloader = DataLoader(trainDataset(os.path.join(dataset, 'train'),
                                                transform=transforms.Compose([
                                                # transforms.ColorJitter(hue=0.1),
                                                                            transforms.Resize([input_size, input_size]),
                                                                            transforms.RandomHorizontalFlip(p=0.5), 
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                                 std=[0.229, 0.224, 0.225])])
                                                ),
                                                                      
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=True)

    
    return train_dataloader



# test-train pair 
def test_dataloader(dataset, input_size=64, batch_size=16, num_workers=0):
    
    
    test1_dataloader = DataLoader(test1Dataset(os.path.join(dataset, 'test'),
                                                transform=transforms.Compose([
                                                                            transforms.Resize([input_size, input_size]),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                                 std=[0.229, 0.224, 0.225])])
                                                ),
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                pin_memory=True)
    test2_dataloader = DataLoader(test2Dataset(os.path.join(dataset, 'train'),
                                                transform=transforms.Compose([
                                                                            transforms.Resize([input_size, input_size]),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                                 std=[0.229, 0.224, 0.225])])
                                                ),
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                pin_memory=True)
    return test1_dataloader, test2_dataloader




class trainDataset(Dataset):
    
    def read_data_set(self):
        all_img_files = []
        all_labels = []
        class_names = os.walk(self.data_set_path).__next__()[1]
        
        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]
            
            for img_file in img_files: 
                img_file = os.path.join(img_dir, img_file)
                img = cv2.imread(img_file, flags=cv2.IMREAD_COLOR)
                
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)
        #print(all_img_files)            
        return all_img_files, all_labels, len(all_img_files), len(class_names)

    
    def __init__(self, dataset_path, transform=None):
        self.data_set_path = dataset_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transform = transform

    
    def __len__(self):
        return self.length


    def __getitem__(self, idx):
      

        inc = random.randrange(1, self.length)
        img1_dir = self.image_files_path[idx]
        img2_dir = self.image_files_path[int((idx+inc)%self.length)]
        img1 = Image.open(img1_dir)
        img2 = Image.open(img2_dir)
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")


        label = 1 if img1_dir.split("\\")[2] == img2_dir.split("\\")[2] else 0
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        x1 = np.array(img1, 'float32')
        x2 = np.array(img2, 'float32')
        y = np.array(label, 'int64')

        return torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(y)
        

class test1Dataset(Dataset):
    
    def read_data_set(self):
        all_img_files = []
        file_names = os.walk(self.data_set_path).__next__()[1]
        
        for index, img_file in enumerate(file_names):
            img = cv2.imread(img_file, flags=cv2.IMREAD_COLOR)
    
            if img is not None:
                all_img_files.append(img_file)
                    
        return all_img_files, len(file_names)

    
    def __init__(self, dataset_path, transform=None):
        self.data_set_path = dataset_path
        self.image_files_path, self.length = self.read_data_set()
        self.transform = transform

    
    def __len__(self):
        return  self.length

    def __getitem__(self, idx):

        image = Image.open(self.image_files_path[idx])
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        x = np.array(image, 'float32')
      
        return torch.from_numpy(x)



class test2Dataset(Dataset):
    
    def read_data_set(self):
        all_img_files = []
        all_labels = []
        class_names = os.walk(self.data_set_path).__next__()[1]
        
        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]
            
            for img_file in img_files: 
                img_file = os.path.join(img_dir, img_file)
                img = cv2.imread(img_file, flags=cv2.IMREAD_COLOR)
                
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)
        #print(all_img_files)            
        return all_img_files, all_labels, len(all_img_files), len(class_names)

    
    def __init__(self, dataset_path, transform=None):
        self.data_set_path = dataset_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transform = transform

    
    def __len__(self):
        return self.length


    def __getitem__(self, idx):
      

        img1_dir = self.image_files_path[idx]
        img1 = Image.open(img1_dir)
        img1 = img1.convert("RGB")
        index = img1_dir.split('\\')[2]
        print(img1_dir, '-> ', index)

        if self.transform:
            img1 = self.transform(img1)


        x = np.array(img1, 'float32')
        return torch.from_numpy(x), index