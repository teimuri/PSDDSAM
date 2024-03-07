from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils import   create_prompt_yours , create_prompt
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
from torch.nn.functional import threshold, normalize
import torchvision.transforms.functional as TF



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
def save_img(img, dir):
    img = img.clone().cpu().numpy() 

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

slice_per_image = 1
image_size = 1024
class panc_sam(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        #Promptless
        sam = torch.load("sam_tuned_save.pth").sam
        
        self.prompt_encoder = sam.prompt_encoder
        
        self.mask_decoder = sam.mask_decoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
            
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        
        #with Prompt
        sam=sam_model_registry[model_type](checkpoint=checkpoint)

        # sam = torch.load(
        #     "sam_tuned_save.pth"
        # ).sam
        self.image_encoder = sam.image_encoder
        self.prompt_encoder2 = sam.prompt_encoder
        self.mask_decoder2 = sam.mask_decoder
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        for param in self.prompt_encoder2.parameters():
            param.requires_grad = False

            

        

    def forward(self, input_images,box=None):
        

        # input_images = torch.stack([x["image"] for x in batched_input], dim=0)
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
            
            # points, point_labels = create_prompt((low_res_masks > 0).float())
            points, point_labels = create_prompt_yours(low_res_masks)
            
            
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
    
test_dataset = PanDataset(
    [
        
     "your data images address"
    # "your data images address"   
     ],
    [
     "your data  labels address"
    #   "your data  labels address"
     ],
        
    [["NIH_PNG",1]],

    image_size,
    
    slice_per_image=slice_per_image,
    train=False,
) 
device = "cuda:0"

x = torch.load('sam_tuned_save.pth', map_location='cpu')
# raise ValueError(x)
x.to(device)
# num_samples = 0
# counterb=0
# index=0
# for image, label in tqdm(test_dataset, total=len(test_dataset)):
#     num_samples += 1
#     counterb += 1
#     index += 1
#     image = image.to(device)
#     label = label.to(device).float()
    
#     ############################model and dice########################################
#     box = torch.tensor([[200, 200, 750, 800]]).to(device)
#     low_res_masks_main,low_res_masks_prompt = x(image,box)

def process_model(main_model , test_dataset, train=0, save_output=0):
    epoch_losses = []
    results=[]
    results_prompt=[]
    index = 0
    results = torch.zeros((2, 0, 256, 256))
    results_prompt = torch.zeros((2, 0, 256, 256))
    
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
    for image, label , raw_data in tqdm(test_dataset, total=len(test_dataset)):
        # raise ValueError(image.shape , label.shape , raw_data.shape)
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

        # binary_mask = normalize(threshold(low_res_masks_main, 0.0,0))
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
        binary_mask = normalize(threshold(low_res_masks_main, 0.0,0))
        binary_mask_mask = normalize(threshold(low_res_masks_prompt, 0.0,0))
        

        
        ###################################
    
        result = torch.cat(
            (
                low_res_masks_main[0].detach().cpu().reshape(1, 1, 256, 256),
                binary_mask[0].detach().cpu().reshape(1, 1, 256, 256),
            ),
            dim=0,
        )
        results = torch.cat((results, result), dim=1)
        
        result_prompt = torch.cat(
            (
                low_res_masks_prompt[0].detach().cpu().reshape(1, 1, 256, 256),
                binary_mask_mask[0].detach().cpu().reshape(1, 1, 256, 256),
            ),
            dim=0,
        )
        results_prompt = torch.cat((results_prompt, result_prompt), dim=1)
        # if counterb == len(test_dataset)-5:
        #     break
        if counterb == 200:
            break
        # elif counterb == sample_size and not train:
        #     break

    return epoch_losses, results,results_prompt, average_dice,average_precision ,average_recall, average_jaccard,average_dice_main,average_precision_main,average_recall_main,average_jaccard_main


print("Testing:")
test_epoch_losses, epoch_results , results_prompt, average_dice_test,average_precision ,average_recall, average_jaccard,average_dice_test_main,average_precision_main,average_recall_main,average_jaccard_main = process_model(
    x,test_dataset
)
# raise ValueError(len(epoch_results[0]))
train_losses = []
train_epochs = []
test_losses = []
test_epochs = []
dice = []
dice_main = []
dice_test = []
dice_test_main =[]
results = []
index = 0
##############################save image#########################################
for image, label , raw_data in tqdm(test_dataset):
    
    if index < 200:
        if not os.path.exists(f" your result_img/batch_{index}"):
            os.mkdir(f" your result_img/batch_{index}")

        save_img(
            image[0],
            f" your result_img/batch_{index}/img.png",
        )
        tensor_raw = torch.tensor(raw_data)
        save_img(
            tensor_raw.T,
            f" your result_img/batch_{index}/raw_img.png",
        )
        
        model_result_resized = TF.resize(epoch_results, size=(1024, 1024))
        result_canvas = torch.zeros_like(image[0])
        result_canvas[1] = label[0][0]
        result_canvas[0] = model_result_resized[1, index]
        blended_result = 0.2 * image[0] + 0.5 * result_canvas
        
        ###################################################################
        
        model_result_resized_prompt = TF.resize(results_prompt, size=(1024, 1024))
        result_canvas_prompt = torch.zeros_like(image[0])
        result_canvas_prompt[1] = label[0][0]
        # raise ValueError(model_result_resized_prompt.shape ,model_result_resized.shape )
        result_canvas_prompt[0] = model_result_resized_prompt[1, index]
        blended_result_prompt = 0.2 * image[0] + 0.5 * result_canvas_prompt
        
        

        save_img(blended_result, f" your result_img/batch_{index}/comb.png")
        save_img(blended_result_prompt, f" your result_img/batch_{index}/comb_prompt.png")
        
        
        save_img(
        epoch_results[1, index].clone(),f" your result_img/batch_{index}/modelresult.png",
        )
        save_img(
                epoch_results[0, index].clone(),f" your result_img/batch_{index}/prob_epoch_{index}.png",)
        

    index += 1
    if index == 200:
        break


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

