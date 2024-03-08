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
from utils import create_prompt_simple
from pre_processer import PreProcessing
from tqdm import tqdm
from args import get_arguments


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

    img_enhanced -= torch.min(img_enhanced)
    img_max = torch.max(img_enhanced)
    if img_max > 0:
        img_enhanced = img_enhanced / img_max
        # raise ValueError(img_max)
        img_enhanced = img_enhanced.unsqueeze(1)
        img_enhanced = img_enhanced.unsqueeze(1)
        img_enhanced = PreProcessing.CLAHE(
            img_enhanced, clip_limit=9.0, grid_size=(4, 4)
        )
        raise ValueError(img_enhanced.shape)
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
        dirs,
        datasets,
        target_image_size,
        slice_per_image,
        split_ratio=0.9,
        train=True,
        val=False,
        augmentation=None,
    ):
        self.data_set_names = []
        self.labels_path = []
        self.images_path = []
        self.embedds_path = []
        self.labels_indexes = []
        self.individual_index = []
        for dir, dataset_name in zip(dirs, datasets):
            labels_dir = dir + "/labels"
            npy_files = [file for file in os.listdir(labels_dir)]
            items_label = sorted(
                npy_files,
                key=lambda x: (
                    int(x.split("_")[2].split(".")[0]),
                    int(x.split("_")[1]),
                ),
            )

            images_dir = dir + "/images"
            npy_files = [file for file in os.listdir(images_dir)]
            items_image = sorted(
                npy_files,
                key=lambda x: (
                    int(x.split("_")[2].split(".")[0]),
                    int(x.split("_")[1]),
                ),
            )
            try:
                embedds_dir = dir + "/embeddings"
                npy_files = [file for file in os.listdir(embedds_dir)]
                items_embedds = sorted(
                    npy_files,
                    key=lambda x: (
                        int(x.split("_")[2].split(".")[0]),
                        int(x.split("_")[1]),
                    ),
                )
                self.embedds_path.extend(
                    [os.path.join(embedds_dir, item) for item in items_embedds]
                )
            except:
                a = 1

            # raise ValueError(items_label[990].split('_')[2].split('.')[0])
            subject_indexes = set()
            for item in items_label:
                subject_indexes.add(int(item.split("_")[2].split(".")[0]))
            indexes = list(subject_indexes)

            self.labels_indexes.extend(indexes)

            self.individual_index.extend(
                [int(item.split("_")[2].split(".")[0]) for item in items_label]
            )

            self.data_set_names.extend([dataset_name[0] for _ in items_label])

            self.labels_path.extend(
                [os.path.join(labels_dir, item) for item in items_label]
            )
            self.images_path.extend(
                [os.path.join(images_dir, item) for item in items_image]
            )

        self.target_image_size = target_image_size
        self.datasets = datasets
        self.slice_per_image = slice_per_image
        self.augmentation = augmentation
        self.individual_index = torch.tensor(self.individual_index)
        if val:
            self.labels_indexes=self.labels_indexes[int(split_ratio*len(self.labels_indexes)):]
        elif train:
            self.labels_indexes=self.labels_indexes[:int(split_ratio*len(self.labels_indexes))]
        

    def __getitem__(self, idx):

        indexes = (self.individual_index == self.labels_indexes[idx]).nonzero()
        
        images_list = []
        labels_list = []
        batched_input = []
        for index in indexes:

            data = np.load(self.images_path[index])
            embedd = np.load(self.embedds_path[index])

            labels = np.load(self.labels_path[index])

            if self.data_set_names[index] == "NIH_PNG":
                x = data.T
                y = rearrange(labels.T, "H W -> 1 H W")
                y = (y == 1).astype(np.uint8)

            elif self.data_set_names[index] == "Abdment1k-npy":
                x = data

                y = rearrange(labels, "H W -> 1 H W")
                y = (y == 4).astype(np.uint8)
            else:
                raise ValueError("Incorect dataset name")

            x = torch.tensor(x)
            
            embedd = torch.tensor(embedd)
            y = torch.tensor(y)


            current_image_size = y.shape[-1]

            points, point_labels = create_prompt_simple(y[:, ::2, ::2].squeeze(1).float())
            
            points *= self.target_image_size // y[:, ::2, ::2].shape[-1]
            
            y = F.interpolate(y.unsqueeze(1), size=self.target_image_size)
            batched_input.append(
                {
                    "image_embedd": embedd,
                    "image": x,
                    "label": y,
                    "point_coords": points[0],
                    "point_labels": point_labels[0],
                    "original_size": (1024, 1024),
                },
            )


        return batched_input

    def collate_fn(self, data):
        batched_input = zip(*data)


        return data

    def __len__(self):
        return len(self.labels_indexes)

  




