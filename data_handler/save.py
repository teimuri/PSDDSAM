import matplotlib.pyplot as plt
import os
import numpy as np
import random
from segment_anything.utils.transforms import ResizeLongestSide
from einops import rearrange
import torch

import os
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader
from time import time
import torch.nn.functional as F
import cv2


    
# def preprocess(image_paths, label_paths):
#     preprocessed_images = []
#     preprocessed_labels = []
#     for image_path, label_path in zip(image_paths, label_paths):
#         # Load image and label from paths
#         image = plt.imread(image_path)
#         label = plt.imread(label_path)
        
#         # Perform your preprocessing steps here
#         # ...
        
#         preprocessed_images.append(image)
#         preprocessed_labels.append(label)
    
#     return preprocessed_images, preprocessed_labels


class PanDataset:
    def __init__(self, images_dir, labels_dir, slice_per_image, train=True):
        #for Abdonomial
        self.images_path = sorted([os.path.join(images_dir, item[:item.rindex('.')] + '_0000.npz') for item in os.listdir(labels_dir) if item.endswith('.npz') and not item.startswith('.')])
        self.labels_path = sorted([os.path.join(labels_dir, item) for item in os.listdir(labels_dir) if item.endswith('.npz') and not item.startswith('.')])
        #for NIH
        # self.images_path = sorted([os.path.join(images_dir, item) for item in os.listdir(labels_dir) if item.endswith('.npy')])
        # self.labels_path = sorted([os.path.join(labels_dir, item) for item in os.listdir(labels_dir) if item.endswith('.npy')])
        

        N = len(self.images_path)
        n = int(N * 0.8)
        self.train = train 
        self.slice_per_image = slice_per_image
        
        if train:
            
            self.labels_path = self.labels_path[:n]
            self.images_path = self.images_path[:n]
            
        else:
            self.labels_path = self.labels_path[n:]
            self.images_path = self.images_path[n:]            


        
    def __getitem__(self, idx):
        now = time()
        # for abdoment
        data = np.load(self.images_path[idx])['arr_0']
        labels = np.load(self.labels_path[idx])['arr_0']
        #for nih
        # data = np.load(self.images_path[idx])
        # labels = np.load(self.labels_path[idx])
        H, W, C = data.shape
        positive_slices = np.any(labels == 1, axis=(0, 1))
        # print("Load from file time = ", time() - now)
        now = time()

        # Find the first and last positive slices
        first_positive_slice = np.argmax(positive_slices)
        last_positive_slice = labels.shape[2] - np.argmax(positive_slices[::-1]) - 1
        dist=last_positive_slice-first_positive_slice

        if self.train:
            
            save_dir = args.images_dir # data address here
            labels_save_dir = args.images_dir #  label address here
        else :
            save_dir = args.iamges_dir  # data address here
            labels_save_dir = args.labels_dir  # label address here
            
        j=0
        for j in range(1):
            slice = range(len(labels[0,0,:]))
            # raise ValueError(labels.shape)
            image_paths = []
            label_paths = []

            for i, slc_idx in enumerate(slice):
                # Saving Image Slices
                image_array = data[:, :, slc_idx]
                
                # Resize the array to 512x512
                resized_image_array = cv2.resize(image_array, (512, 512))
                
                min_val = resized_image_array.min()
                max_val = resized_image_array.max()
                normalized_image_array = ((resized_image_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                image_paths.append(f"slice_{i}_{idx}.npy")
                print(i)
                if normalized_image_array.max()>0:
                    np.save(os.path.join(save_dir, image_paths[-1]), normalized_image_array)
                    
                    # Saving Corresponding Label Slices
                    label_array = labels[:, :, slc_idx]
                    
                    # Resize the array to 512x512
                    resized_label_array = cv2.resize(label_array, (512, 512))
                    
                    min_val = resized_label_array.min()
                    max_val = resized_label_array.max()
                    # raise ValueError(np.unique(resized_label_array))
                    # normalized_label_array = ((resized_label_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    label_paths.append(f"label_{i}_{idx}.npy")
                    np.save(os.path.join(labels_save_dir, label_paths[-1]), resized_label_array)
                
        

        
        
        return data
    
        

    
    def collate_fn(self, data):
        
        return data

    def __len__(self):
        return len(self.images_path)

if __name__ == '__main__':
    model_type = 'vit_b'
    batch_size = 4
    num_workers = 4
    slice_per_image = 1
    
    dataset = PanDataset('../../Data/AbdomenCT-1K/numpy/images', '../../Data/AbdomenCT-1K/numpy/labels',
                         slice_per_image=slice_per_image)
    # x, y = dataset[7]
    # # print(x.shape, y.shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=True, drop_last=False, num_workers=num_workers)

    now = time()
    for data in dataloader:
        # pass
        # print(images.shape, labels.shape)
        continue
    dataset = PanDataset('../../Data/AbdomenCT-1K/numpy/images', '../../Data/AbdomenCT-1K/numpy/labels', 
                         train = False , slice_per_image=slice_per_image)
    # x, y = dataset[7]
    # # print(x.shape, y.shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=True, drop_last=False, num_workers=num_workers)

    now = time()
    for data in dataloader:
        # pass
        # print(images.shape, labels.shape)
        continue
    
