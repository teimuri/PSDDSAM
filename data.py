import matplotlib.pyplot as plt
import os
import numpy as np
import random
from segment_anything.utils.transforms import ResizeLongestSide
from einops import rearrange
import torch
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader
from time import time
import torch.nn.functional as F
import cv2
from PIL import Image
import cv2

from pre_processer import PreProcessing


def apply_median_filter(input_matrix, kernel_size=5, sigma=0):
    # Apply the Gaussian filter
    filtered_matrix = cv2.medianBlur(input_matrix.astype(np.uint8), kernel_size)

    return filtered_matrix.astype(np.float32)


def apply_guassain_filter(input_matrix, kernel_size=(7, 7), sigma=0):
    smoothed_matrix = cv2.blur(input_matrix, kernel_size)

    return smoothed_matrix.astype(np.float32)


def img_enhance(img2, over_coef=0.8, under_coef=0.7):
    img2 = apply_median_filter(img2)
    img_blure = apply_guassain_filter(img2)

    img2 = img2 - 0.8 * img_blure

    img_mean = np.mean(img2, axis=(1, 2))

    img_max = np.amax(img2, axis=(1, 2))

    val = (img_max - img_mean) * over_coef + img_mean

    img2 = (img2 < img_mean * under_coef).astype(np.float32) * img_mean * under_coef + (
        (img2 >= img_mean * under_coef).astype(np.float32)
    ) * img2

    img2 = (img2 <= val).astype(np.float32) * img2 + (img2 > val).astype(
        np.float32
    ) * val

    return img2


def normalize_and_pad(x, img_size):
    """Normalize pixel values and pad to a square input."""

    pixel_mean = torch.tensor([[[[123.675]], [[116.28]], [[103.53]]]])
    pixel_std = torch.tensor([[[[58.395]], [[57.12]], [[57.375]]]])

    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def preprocess(img_enhanced, img_enhance_times=1, over_coef=0.4, under_coef=0.5):
    # img_enhanced = img_enhanced+0.1

    img_enhanced -= torch.amin(img_enhanced, dim=(1, 2), keepdim=True)
    img_max = torch.amax(img_enhanced, axis=(1, 2), keepdims=True)
    img_max[img_max == 0] = 1
    img_enhanced = img_enhanced / img_max
    # raise ValueError(img_max)
    img_enhanced = img_enhanced.unsqueeze(1)

    img_enhanced = PreProcessing.CLAHE(img_enhanced, clip_limit=9.0, grid_size=(4, 4))
    img_enhanced = img_enhanced[0]

    # for i in range(img_enhance_times):
    #     img_enhanced=img_enhance(img_enhanced.astype(np.float32), over_coef=over_coef,under_coef=under_coef)

    img_enhanced -= torch.amin(img_enhanced, dim=(1, 2), keepdim=True)
    larg_imag = (
        img_enhanced / torch.amax(img_enhanced, axis=(1, 2), keepdims=True) * 255
    ).type(torch.uint8)

    return larg_imag


def prepare(larg_imag, target_image_size):
    # larg_imag = 255 - larg_imag
    larg_imag = rearrange(larg_imag, "S H W -> S 1 H W")
    larg_imag = torch.tensor(
        np.concatenate([larg_imag, larg_imag, larg_imag], axis=1)
    ).float()
    transform = ResizeLongestSide(target_image_size)
    larg_imag = transform.apply_image_torch(larg_imag)
    larg_imag = normalize_and_pad(larg_imag, target_image_size)
    return larg_imag


def process_single_image(image_path, target_image_size):
    # Load the image
    if image_path.endswith(".png") or image_path.endswith(".jpg"):
        data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).squeeze()
    else:
        data = np.load(image_path)
    x = rearrange(data, "H W -> 1 H W")
    x = torch.tensor(x)

    # Apply preprocessing
    x = preprocess(x)
    x = prepare(x, target_image_size)

    return x


class PanDataset:
    def __init__(
        self,
        images_dirs,
        labels_dirs,
        datasets,
        target_image_size,
        slice_per_image,
        train=True,
        augmentation=None,
    ):
        self.data_set_names = []
        self.labels_path = []
        self.images_path = []
        for labels_dir, images_dir, dataset_name in zip(
            labels_dirs, images_dirs, datasets
        ):
            self.data_set_names.extend(
                sorted([dataset_name[0] for _ in os.listdir(labels_dir)])
            )

            self.labels_path.extend(
                sorted([os.path.join(labels_dir, item) for item in os.listdir(labels_dir)])
            )
            self.images_path.extend(
                sorted([os.path.join(images_dir, item) for item in os.listdir(images_dir)])
            )

        self.target_image_size = target_image_size
        self.datasets = datasets
        self.slice_per_image = slice_per_image
        self.augmentation = augmentation

    def __getitem__(self, idx):
        data = np.load(self.images_path[idx])
        raw_data = data

        labels = np.load(self.labels_path[idx])
        

        if self.data_set_names[idx] == "NIH_PNG":
            x = rearrange(data.T, "H W -> 1 H W")

            y = rearrange(labels.T, "H W -> 1 H W")
            y = (y == 1).astype(np.uint8)
            
            
            

        elif self.data_set_names[idx] == "Abdment1kPNG":
            x = rearrange(data, "H W -> 1 H W")

            y = rearrange(labels, "H W -> 1 H W")
            y = (y == 4).astype(np.uint8)
        else:
            raise ValueError("Incorect dataset name")

        x = torch.tensor(x)
        y = torch.tensor(y)
        x = preprocess(x)
        


        x, y = self.apply_augmentation(x.numpy(), y.numpy())

        y = F.interpolate(y.unsqueeze(1), size=self.target_image_size)

        x = prepare(x, self.target_image_size)
        return x, y ,raw_data

    def collate_fn(self, data):
        images, labels , raw_data = zip(*data)
        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        # raw_data = torch.cat(raw_data, dim=0)
        
        return images, labels , raw_data 

    def __len__(self):
        return len(self.images_path)

    def apply_augmentation(self, image, label):
        if self.augmentation:
            # If image and label are tensors, convert them to numpy arrays
            # raise ValueError(label.shape)
            augmented = self.augmentation(image=image[0], mask=label[0])

            image = torch.tensor(augmented["image"])
            label = torch.tensor(augmented["mask"])

            # You might want to convert back to torch.Tensor after the transformation
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)

        else:
            image = torch.Tensor(image)
            label = torch.Tensor(label)

        return image, label


import albumentations as A

if __name__ == "__main__":
    model_type = "vit_h"
    batch_size = 4
    num_workers = 4
    slice_per_image = 1
    image_size = 1024

    checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    panc_sam_instance = sam_model_registry[model_type](checkpoint=checkpoint)

    augmentation = A.Compose(
        [
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomResizedCrop(1024, 1024, scale=(0.9, 1.0), p=1),
        ]
    )
    train_dataset = PanDataset(
        "your images addres here",
        "your labeles addres here",
        image_size,
        slice_per_image=slice_per_image,
        train=True,
        augmentation=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    # x, y = dataset[7]
    # print(x.shape, y.shape)

    now = time()
    for images, labels in train_loader:
        # pass
        image_numpy = images[0].permute(1, 2, 0).cpu().numpy()

        # Ensure that the values are in the correct range [0, 255] and cast to uint8
        image_numpy = (image_numpy * 255).astype(np.uint8)

        # Save the image using OpenCV
        cv2.imwrite("image2.png", image_numpy[:, :, 1])

        break

    # print((time() - now) / batch_size / slice_per_image)
