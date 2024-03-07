from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils import create_prompt
from collections import defaultdict
import torchvision.transforms as transforms
import torch
from torch import nn
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide
import albumentations as A
from albumentations.pytorch import ToTensorV2
from einops import rearrange
import random
from tqdm import tqdm
from time import sleep
from data import *
from time import time
from PIL import Image
from sklearn.model_selection import KFold
from shutil import copyfile

# import wandb_handler


def save_img(img, dir):
    img = img.clone().cpu().numpy() + 100

    if len(img.shape) == 3:
        img = rearrange(img, "c h w -> h w c")
        img_min = np.amin(img, axis=(0, 1), keepdims=True)
        img = img - img_min

        img_max = np.amax(img, axis=(0, 1), keepdims=True)
        img = (img / img_max * 255).astype(np.uint8)
        img = Image.fromarray(img)

    else:
        img_min = img.min()
        img = img - img_min
        img_max = img.max()
        if img_max != 0:
            img = img / img_max * 255
        img = Image.fromarray(img).convert("L")

    img.save(dir)



class loss_fn(torch.nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, epsilon=1e-5):
        super(loss_fn, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def dice_loss(self, logits, gt, eps=1):

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        gt = gt.view(-1)

        intersection = (probs * gt).sum()

        dice_coeff = (2.0 * intersection + eps) / (probs.sum() + gt.sum() + eps)

        loss = 1 - dice_coeff
        return loss

    def focal_loss(self, logits, gt, gamma=2):
        logits = logits.reshape(-1, 1)
        gt = gt.reshape(-1, 1)
        logits = torch.cat((1 - logits, logits), dim=1)

        probs = torch.sigmoid(logits)
        pt = probs.gather(1, gt.long())

        modulating_factor = (1 - pt) ** gamma
    
        focal_loss = -modulating_factor * torch.log(pt + 1e-12)

        loss = focal_loss.mean()
        return loss  # Store as a Python number to save memory

    def forward(self, logits, target):
        logits = logits.squeeze(1)
        target = target.squeeze(1)
        # Dice Loss
        # prob = F.softmax(logits, dim=1)[:, 1, ...]

        dice_loss = self.dice_loss(logits, target)

        # Focal Loss
        focal_loss = self.focal_loss(logits, target.squeeze(-1))
        alpha = 20.0
        # Combined Loss
        combined_loss = alpha * focal_loss + dice_loss
        return combined_loss


def img_enhance(img2, coef=0.2):
    img_mean = np.mean(img2)
    img_max = np.max(img2)
    val = (img_max - img_mean) * coef + img_mean
    img2[img2 < img_mean * 0.7] = img_mean * 0.7
    img2[img2 > val] = val
    return img2


def dice_coefficient(logits, gt):
    
    eps=1
    binary_mask = logits>0
    
    intersection = (binary_mask * gt).sum(dim=(-2,-1))
    dice_scores = (2.0 * intersection + eps) / (binary_mask.sum(dim=(-2,-1)) + gt.sum(dim=(-2,-1)) + eps)
    
    return dice_scores.mean()


def calculate_recall(pred, target):
    smooth = 1
    batch_size = pred.shape[0]
    recall_scores = []
    binary_mask = pred>0

    for i in range(batch_size):
        true_positive = ((binary_mask[i] == 1) & (target[i] == 1)).sum().item()
        false_negative = ((binary_mask[i] == 0) & (target[i] == 1)).sum().item()
        recall = (true_positive + smooth) / ((true_positive + false_negative) + smooth)
        recall_scores.append(recall)

    return sum(recall_scores) / len(recall_scores)

def calculate_precision(pred, target):
    smooth = 1
    batch_size = pred.shape[0]
    precision_scores = []
    binary_mask = pred>0

    for i in range(batch_size):
        true_positive = ((binary_mask[i] == 1) & (target[i] == 1)).sum().item()
        false_positive = ((binary_mask[i] == 1) & (target[i] == 0)).sum().item()
        precision = (true_positive + smooth) / ((true_positive + false_positive) + smooth)
        precision_scores.append(precision)

    return sum(precision_scores) / len(precision_scores)

def calculate_jaccard(pred, target):
    smooth = 1
    batch_size = pred.shape[0]
    jaccard_scores = []
    binary_mask = pred>0
    

    for i in range(batch_size):
        true_positive = ((binary_mask[i] == 1) & (target[i] == 1)).sum().item()
        false_positive = ((binary_mask[i] == 1) & (target[i] == 0)).sum().item()
        false_negative = ((binary_mask[i] == 0) & (target[i] == 1)).sum().item()
        jaccard = (true_positive + smooth) / (true_positive + false_positive + false_negative + smooth)
        jaccard_scores.append(jaccard)

    return sum(jaccard_scores) / len(jaccard_scores)

def calculate_specificity(pred, target):
    smooth = 1
    batch_size = pred.shape[0]
    specificity_scores = []
    binary_mask = pred>0
    

    for i in range(batch_size):
        true_negative = ((binary_mask[i] == 0) & (target[i] == 0)).sum().item()
        false_positive = ((binary_mask[i] == 1) & (target[i] == 0)).sum().item()
        specificity = (true_negative + smooth) / (true_negative + false_positive + smooth)
        specificity_scores.append(specificity)

    return sum(specificity_scores) / len(specificity_scores)

def what_the_f(low_res_masks,label):
            
    low_res_label = F.interpolate(label, low_res_masks.shape[-2:])
    dice = dice_coefficient(
        low_res_masks, low_res_label
    )
    recall=calculate_recall(low_res_masks, low_res_label)
    precision =calculate_precision(low_res_masks, low_res_label)
    jaccard = calculate_jaccard(low_res_masks, low_res_label)
    
    return dice , precision , recall , jaccard 

accumaltive_batch_size = 8
batch_size = 1
num_workers = 2
slice_per_image = 1
num_epochs = 40
sample_size = 3660
# sample_size = 43300
# image_size=sam_model.image_encoder.img_size
image_size = 1024
exp_id = 0
found = 0



layer_n = 4
L = layer_n
a = np.full(L, layer_n)
params = {"M": 255, "a": a, "p": 0.35}


model_type = "vit_h"
checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
device = "cuda:0"


from segment_anything import SamPredictor, sam_model_registry



##################################main model#######################################



class panc_sam(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        #Promptless
        sam = torch.load("weightes/sam_tuned_save.pth").sam
        
        self.prompt_encoder = sam.prompt_encoder
        
        self.mask_decoder = sam.mask_decoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
            
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        
        #with Prompt
        sam = torch.load(
            "weightes/sam_tuned_save.pth"
        ).sam
        self.image_encoder = sam.image_encoder
        self.prompt_encoder2 = sam.prompt_encoder
        self.mask_decoder2 = sam.mask_decoder
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        for param in self.prompt_encoder2.parameters():
            param.requires_grad = False
            

        

    def forward(self, input_images,box=None):
        

        # input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        # raise ValueError(input_images.shape)
        with torch.no_grad():
            image_embeddings = self.image_encoder(input_images).detach()

            
        outputs_prompt = []
        outputs = []
        
        for curr_embedding in image_embeddings:
            
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
                )
            
            low_res_masks, _ = self.mask_decoder(
                image_embeddings=curr_embedding,
                image_pe=self.prompt_encoder.get_dense_pe().detach(),
                sparse_prompt_embeddings=sparse_embeddings.detach(),
                dense_prompt_embeddings=dense_embeddings.detach(),
                multimask_output=False,
            )
            outputs_prompt.append(low_res_masks)
            # raise ValueError(low_res_masks)
            # points, point_labels = create_prompt((low_res_masks > 0).float())
            points, point_labels = create_prompt(low_res_masks)
            
            
            points = points * 4
            points = (points, point_labels)

            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder2(
                     points=points,
                     boxes=None,
                     masks=None,
                 )
            
            low_res_masks, _ = self.mask_decoder2(
                image_embeddings=curr_embedding,
                image_pe=self.prompt_encoder2.get_dense_pe().detach(),
                sparse_prompt_embeddings=sparse_embeddings.detach(),
                dense_prompt_embeddings=dense_embeddings.detach(),
                multimask_output=False,
            )
            
            outputs.append(low_res_masks)
        low_res_masks_promtp = torch.cat(outputs_prompt, dim=0)
        low_res_masks = torch.cat(outputs, dim=0)
        

        return low_res_masks, low_res_masks_promtp
##################################end#######################################

##################################Augmentation#######################################

augmentation = A.Compose(
    [
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
        A.RandomResizedCrop(1024, 1024, scale=(0.9, 1.0), p=1),
        A.HorizontalFlip(p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.5,
        ),
        A.RandomScale(scale_limit=0.3, p=0.5),
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # A.GridDistortion(p=0.5),
    ]
)
##################model load#####################
panc_sam_instance = panc_sam()

# for param in panc_sam_instance_point.parameters():
#     param.requires_grad = False
panc_sam_instance.to(device)
panc_sam_instance.train()


##################load data#######################


test_dataset = PanDataset(
    [args.test_dir],
    [args.test_labels_dir],
        
    [["NIH_PNG",1]],

    image_size,
    
    slice_per_image=slice_per_image,
    train=False,
) 

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=test_dataset.collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)
##################end load data#######################

lr = 1e-4
max_lr = 5e-5
wd = 5e-4

optimizer_main = torch.optim.Adam(
    # parameters,
    list(panc_sam_instance.mask_decoder2.parameters()),
    
    lr=lr,
    weight_decay=wd,
)
scheduler_main = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_main,
    max_lr=max_lr,
    epochs=num_epochs,
    steps_per_epoch=sample_size // (accumaltive_batch_size // batch_size),
)
#####################################################

from statistics import mean

from tqdm import tqdm
from torch.nn.functional import threshold, normalize

loss_function = loss_fn(alpha=0.5, gamma=2.0)
loss_function.to(device)

from time import time
import time as s_time



def process_model(main_model , data_loader, train=0, save_output=0):
    epoch_losses = []
    results=[]
    index = 0
    results = torch.zeros((2, 0, 256, 256))
    #############################
    total_dice = 0.0
    total_precision = 0.0
    total_recall =0.0
    total_jaccard = 0.0
    #############################
    num_samples = 0
    #############################
    total_dice_main =0.0
    total_precision_main = 0.0
    total_recall_main =0.0
    total_jaccard_main = 0.0

    counterb = 0
    for image, label in tqdm(data_loader, total=sample_size):
        num_samples += 1
        counterb += 1
        index += 1
        image = image.to(device)
        label = label.to(device).float()
        
        ############################model and dice########################################
        box = torch.tensor([[200, 200, 750, 800]]).to(device)
        low_res_masks_main,low_res_masks_prompt = main_model(image,box)
        
        low_res_label = F.interpolate(label, low_res_masks_main.shape[-2:]) 
        
        dice_prompt,  precisio_prompt , recall_prompt , jaccard_prompt  = what_the_f(low_res_masks_prompt,low_res_label)
        dice_main , precision_main , recall_main , jaccard_main  = what_the_f(low_res_masks_main,low_res_label)

        binary_mask = normalize(threshold(low_res_masks_main, 0.0,0))
        ##############prompt###############
        total_dice += dice_prompt
        total_precision += precisio_prompt
        total_recall += recall_prompt
        total_jaccard += jaccard_prompt
        average_dice = total_dice / num_samples
        average_precision = total_precision /num_samples
        average_recall = total_recall /num_samples
        average_jaccard = total_jaccard /num_samples
        
        ##############main##################
        total_dice_main+=dice_main
        total_precision_main +=precision_main
        total_recall_main +=recall_main
        total_jaccard_main += jaccard_main
        
        
        average_dice_main = total_dice_main / num_samples
        average_precision_main = total_precision_main /num_samples
        average_recall_main = total_recall_main /num_samples
        average_jaccard_main = total_jaccard_main /num_samples
        
        ###################################
    
        # result = torch.cat(
        #     (
        #         # low_res_masks_main[0].detach().cpu().reshape(1, 1, 256, 256),
        #         binary_mask[0].detach().cpu().reshape(1, 1, 256, 256),
        #     ),
        #     dim=0,
        # )
        # results = torch.cat((results, result), dim=1)

        if counterb == sample_size and train:
            break
        elif counterb == sample_size and not train:
            break

    return epoch_losses, results, average_dice,average_precision ,average_recall, average_jaccard,average_dice_main,average_precision_main,average_recall_main,average_jaccard_main



def train_model( test_loader, K_fold=False, N_fold=7, epoch_num_start=7):
    print("Train model started.")

    test_losses = []
    test_epochs = []
    dice = []
    dice_main = []
    dice_test = []
    dice_test_main =[]
    results = []
    index = 0

    print("Testing:")
    test_epoch_losses, epoch_results, average_dice_test,average_precision ,average_recall, average_jaccard,average_dice_test_main,average_precision_main,average_recall_main,average_jaccard_main = process_model(
        panc_sam_instance,test_loader
    )
    import torchvision.transforms.functional as TF




    dice_test.append(average_dice_test)
    dice_test_main.append(average_dice_test_main)
    print("######################Prompt##########################")
    print(f"Test Dice : {average_dice_test}")
    print(f"Test presision : {average_precision}")
    print(f"Test recall : {average_recall}")
    print(f"Test jaccard : {average_jaccard}")
    
    print("######################Main##########################")
    print(f"Test Dice main : {average_dice_test_main}")
    print(f"Test presision main : {average_precision_main}")
    print(f"Test recall main : {average_recall_main}")
    print(f"Test jaccard main : {average_jaccard_main}")
    
    
    # results.append(epoch_results)
    # del epoch_results
    del average_dice_test
            

    # return train_losses,  results


train_model(test_loader)

