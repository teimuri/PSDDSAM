from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# import cv2
from collections import defaultdict
import torchvision.transforms as transforms
import torch
from torch import nn

import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

import random
from tqdm import tqdm
from time import sleep
from data_load_group import *
from time import time
from PIL import Image
from sklearn.model_selection import KFold
from shutil import copyfile
# import monai
from utils import *
from torch.autograd import Variable
from Models.model_handler import panc_sam
from Models.model_handler import UNet3D
from Models.model_handler import Conv3DFilter
from loss import *
from args import get_arguments
from segment_anything import SamPredictor, sam_model_registry
from statistics import mean
from copy import deepcopy
from torch.nn.functional import threshold, normalize
def calculate_recall(pred, target):
    smooth = 1
    batch_size = 1
    recall_scores = []
    
    binary_mask = pred>0

    true_positive = ((pred == 1) & (target == 1)).sum().item()
    false_negative = ((pred == 0) & (target == 1)).sum().item()
    recall = (true_positive + smooth) / ((true_positive + false_negative) + smooth)
    

    
    return recall

def calculate_precision(pred, target):
    smooth = 1
    batch_size = 1
    recall_scores = []
    
    binary_mask = pred>0

    true_positive = ((pred == 1) & (target == 1)).sum().item()
    false_negative = ((pred == 1) & (target == 0)).sum().item()
    recall = (true_positive + smooth) / ((true_positive + false_negative) + smooth)
    

    
    return recall

def calculate_jaccard(pred, target):
    smooth = 1
    batch_size = pred.shape[0]
    jaccard_scores = []
    binary_mask = pred>0
    true_positive = ((pred == 1) & (target == 1)).sum().item()
    false_negative = ((pred == 0) & (target == 1)).sum().item()
    true_positive = ((pred == 1) & (target == 1)).sum().item()
    false_positive = ((pred == 1) & (target == 0)).sum().item()
    jaccard = (true_positive + smooth) / (true_positive + false_positive + false_negative + smooth)
    # jaccard_scores.append(jaccard)
    # for i in range(batch_size):
    #     true_positive = ((binary_mask[i] == 1) & (target[i] == 1)).sum().item()
    #     false_positive = ((binary_mask[i] == 1) & (target[i] == 0)).sum().item()
    #     false_negative = ((binary_mask[i] == 0) & (target[i] == 1)).sum().item()
        

    return jaccard

def save_img(img, dir):
    img = img.clone().cpu().numpy() + 100

    if len(img.shape) == 3:
        img = rearrange(img, "c h w -> h w c")
        img_min = np.amin(img, axis=(0, 1), keepdims=True)
        img = img - img_min

        img_max = np.amax(img, axis=(0, 1), keepdims=True)
        img = (img / img_max * 255).astype(np.uint8)
        # grey_img = Image.fromarray(img[:, :, 0])
        
        img = Image.fromarray(img)

    else:
        img_min = img.min()
        img = img - img_min
        img_max = img.max()
        if img_max != 0:
            img = img / img_max * 255
        
        img = Image.fromarray(img).convert("L")

    img.save(dir)
    
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(2024)


global optimizer
args = get_arguments()


exp_id = 0
found = 0
user_input = args.run_name
while found == 0:
    try:
        os.makedirs(f"exps/{exp_id}-{user_input}")
        found = 1
    except:
        exp_id = exp_id + 1
copyfile(os.path.realpath(__file__), f"exps/{exp_id}-{user_input}/code.py")

augmentation = A.Compose(
    [
        A.Rotate(limit=100, p=0.7),
        A.RandomScale(scale_limit=0.3, p=0.5),
    ]
)
device = "cuda:0"



panc_sam_instance = torch.load(args.model_path)
panc_sam_instance.to(device)

conv3d_instance = UNet3D()
kernel_size = [(1, 5, 5), (5, 5, 5), (5, 5, 5), (5, 5, 5)]

conv3d_instance = Conv3DFilter(
    1,
    5,
    kernel_size,
    np.array(kernel_size) // 2,
    custom_bias=args.custom_bias,
)

conv3d_instance.to(device)
conv3d_instance.train()

train_dataset = PanDataset(
    dirs=[f"{args.train_dir}/train"],
    datasets=[["NIH_PNG", 1]],
    target_image_size=args.image_size,
    slice_per_image=args.slice_per_image,
    train=True,
    val=False,  # Enable validation data splitting
    augmentation=augmentation,
)

val_dataset = PanDataset(
    [f"{args.train_dir}/train"],
    [["NIH_PNG", 1]],
    args.image_size,
    slice_per_image=args.slice_per_image,
    val=True
)

test_dataset = PanDataset(
    [f"{args.test_dir}/test"],
    [["NIH_PNG", 1]],
    args.image_size,
    slice_per_image=args.slice_per_image,
    train=False,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    collate_fn=train_dataset.collate_fn,
    shuffle=True,
    drop_last=False,
    num_workers=args.num_workers,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    collate_fn=val_dataset.collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    collate_fn=test_dataset.collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers,
)


# Set up the optimizer, hyperparameter tuning will improve performance here
lr = args.lr
max_lr = 3e-4
wd = 5e-4


all_parameters = list(conv3d_instance.parameters())

optimizer = torch.optim.Adam(
    all_parameters,
    lr=lr,
    weight_decay=wd,
)


scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.7, verbose=True
)

loss_function = loss_fn(alpha=0.5, gamma=2.0)
loss_function.to(device)

from time import time
import time as s_time

log_file = open(f"exps/{exp_id}-{user_input}/log.txt", "a")


def process_model(data_loader, train=0, save_output=0, epoch=None, scheduler=None):
    epoch_losses = []

    index = 0
    results = []
    dice_sam_lists = []
    dice_sam_prompt_lists = []
    dice_lists = []
    dice_prompt_lists = []

    num_samples = 0

    counterb = 0

    for batch in tqdm(data_loader, total=args.sample_size // args.batch_size):

        for batched_input in batch:
            
            num_samples += 1

            # raise ValueError(len(batched_input))
            low_res_masks = torch.zeros((1, 1, 0, 256, 256))
            # s_time.sleep(0.6)
            counterb += len(batch)

            index += 1
            label = []
            label = [i["label"] for i in batched_input]

            # Only correct if gray scale
            label = torch.cat(label, dim=1)
            # raise ValueError(la)
            label = label.float()

            true_indexes = torch.where((torch.amax(label, dim=(2, 3)) > 0).view(-1))[0]

            low_res_label = F.interpolate(label, low_res_masks.shape[-2:]).to("cuda:0")
            low_res_masks, low_res_masks_promtp = panc_sam_instance(
                batched_input, device
            )
            
            low_res_shape = low_res_masks.shape[-2:]
            low_res_label_prompt=low_res_label
            if train:
                transformed = augmentation(
                    image=low_res_masks_promtp[0].permute(1, 2, 0).cpu().numpy(),
                    mask=low_res_label[0].permute(1, 2, 0).cpu().numpy(),
                )

                low_res_masks_promtp = (
                    torch.tensor(transformed["image"])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )

                low_res_label_prompt = (
                    torch.tensor(transformed["mask"])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )

                transformed = augmentation(
                    image=low_res_masks[0].permute(1, 2, 0).cpu().numpy(),
                    mask=low_res_label[0].permute(1, 2, 0).cpu().numpy(),
                )

                low_res_masks = (
                    torch.tensor(transformed["image"])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )

                low_res_label = (
                    torch.tensor(transformed["mask"])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
                low_res_masks = F.interpolate(low_res_masks, low_res_shape).to(device)
                low_res_label = F.interpolate(low_res_label, low_res_shape).to(device)

                low_res_masks = F.interpolate(low_res_masks, low_res_shape).to(device)
                low_res_label = F.interpolate(low_res_label, low_res_shape).to(device)
                low_res_masks_promtp = F.interpolate(
                    low_res_masks_promtp, low_res_shape
                ).to(device)
                low_res_label_prompt = F.interpolate(
                    low_res_label_prompt, low_res_shape
                ).to(device)
            low_res_masks = low_res_masks.detach()
            low_res_masks_promtp = low_res_masks_promtp.detach()

            dice_sam = dice_coefficient(low_res_masks , low_res_label).detach().cpu()
            dice_sam_prompt = (
                dice_coefficient(low_res_masks_promtp, low_res_label_prompt)
                .detach()
                .cpu()
            )
            low_res_masks_promtp = conv3d_instance(
                low_res_masks_promtp.detach().to(device)
            )

            loss = loss_function(low_res_masks_promtp, low_res_masks_promtp)
            loss /= args.accumulative_batch_size / args.batch_size

            binary_mask = low_res_masks > 0.5
            binary_mask_prompt = low_res_masks_promtp > 0.5

            dice = dice_coefficient(binary_mask, low_res_label).detach().cpu()
            dice_prompt = (
                dice_coefficient(binary_mask_prompt, low_res_label_prompt)
                .detach()
                .cpu()
            )

            dice_sam_lists.append(dice_sam)
            dice_sam_prompt_lists.append(dice_sam_prompt)
            dice_lists.append(dice)
            dice_prompt_lists.append(dice_prompt)

            log_file.flush()
            if train:
                loss.backward()
                if index % (args.accumulative_batch_size / args.batch_size) == 0:

                    optimizer.step()
                    # if epoch==40:
                    #     scheduler.step()
                    optimizer.zero_grad()
                    index = 0

            else:

                result = torch.cat(
                    (
                        low_res_masks[:, ::10].detach().cpu().reshape(1, -1, 256, 256),
                        binary_mask[:, ::10].detach().cpu().reshape(1, -1, 256, 256),
                    ),
                    dim=0,
                )
                results.append(result)

        if index % (args.accumulative_batch_size / args.batch_size) == 0:
            epoch_losses.append(loss.item())
        if counterb == (args.sample_size // args.batch_size) and train:
            break


    return (
        epoch_losses,
        results,
        dice_lists,
        dice_prompt_lists,
        dice_sam_lists,
        dice_sam_prompt_lists,
    )


def train_model(train_loader, val_loader,test_loader, K_fold=False, N_fold=7, epoch_num_start=7):
    global optimizer
    index=0
    if args.inference:
        with torch.no_grad():
            conv = torch.load(f'{args.conv_path}')
        
        recall_list=[]
        percision_list=[]
        jaccard_list=[]
        
        for input in tqdm(test_loader):
            
            
            low_res_masks_sam, low_res_masks_promtp_sam = panc_sam_instance(
                                                    input[0], device
                                                )
            low_res_masks_sam = F.interpolate(low_res_masks_sam, 512).cpu()
            low_res_masks_promtp_sam = F.interpolate(low_res_masks_promtp_sam, 512).cpu()
            low_res_masks_promtp = conv(low_res_masks_promtp_sam.to(device)).detach().cpu()
            
            for slice_id,(batched_input,mask_sam,mask_prompt_sam,mask_prompt) in enumerate(zip(input[0],low_res_masks_sam[0],low_res_masks_promtp_sam[0],low_res_masks_promtp[0])):
                
                if not os.path.exists(f"ims/batch_{index}"):
                    os.mkdir(f"ims/batch_{index}")
                image = batched_input["image"]
                
                
                label = batched_input["label"][0,0,::2,::2].to(bool)
                binary_mask_sam = (mask_sam > 0)
                binary_mask_prompt_sam = (mask_prompt_sam > 0)
                binary_mask_prompt = (mask_prompt > 0.5)
                recall = calculate_recall(label, binary_mask_prompt)
                percision = calculate_precision(label, binary_mask_prompt)
                jaccard = calculate_jaccard(label, binary_mask_prompt)
                percision_list.append(percision)
                recall_list.append(recall)
                jaccard_list.append(jaccard)
                image_mask = image.clone().to(torch.long)
                image_label = image.clone().to(torch.long)
                image_mask[binary_mask_sam]=255
                image_label[label]=255
                save_img(
                    torch.stack((image_mask,image_label,image),dim=0),
                    f"ims/batch_{index}/sam{slice_id}.png",
                )
                
                image_mask = image.clone().to(torch.long)
                image_mask[binary_mask_prompt_sam]=255

                save_img(
                    torch.stack((image_mask,image_label,image),dim=0),
                    f"ims/batch_{index}/sam_prompt{slice_id}.png",
                )
                
                image_mask = image.clone().to(torch.long)
                image_mask[binary_mask_prompt]=255
                save_img(
                    torch.stack((image_mask,image_label,image),dim=0),
                    f"ims/batch_{index}/prompt_{slice_id}.png",
                )
            print(f'Recall={np.mean(recall_list)}')
            print(f'Percision={np.mean(percision_list)}')
            print(f'Jaccard={np.mean(jaccard_list)}')
            index += 1
        print(f'Recall={np.mean(recall_list)}')
        print(f'Percision={np.mean(percision_list)}')
        print(f'Jaccard={np.mean(jaccard_list)}')
    else:
        
        print("Train model started.")

        train_losses = []
        train_epochs = []
        val_losses = []
        val_epochs = []
        dice = []
        dice_val = []
        results = []
        last_best_dice=0
        for epoch in range(args.num_epochs):

            print(f"=====================EPOCH: {epoch + 1}=====================")
            log_file.write(f"=====================EPOCH: {epoch + 1}===================\n")

            print("Training:")
            (
                train_epoch_losses,
                results,
                dice_list,
                dice_prompt_list,
                dice_sam_list,
                dice_sam_prompt_list,
            ) = process_model(train_loader, train=1, epoch=epoch, scheduler=scheduler)

            dice_mean = np.mean(dice_list)
            dice_prompt_mean = np.mean(dice_prompt_list)
            dice_sam_mean = np.mean(dice_sam_list)
            dice_sam_prompt_mean = np.mean(dice_sam_prompt_list)

            print("Validating:")
            (
                _,
                _,
                val_dice_list,
                val_dice_prompt_list,
                val_dice_sam_list,
                val_dice_sam_prompt_list,
            ) = process_model(val_loader)
            val_dice_mean = np.mean(val_dice_list)
            val_dice_prompt_mean = np.mean(val_dice_prompt_list)
            val_dice_sam_mean = np.mean(val_dice_sam_list)
            val_dice_sam_prompt_mean = np.mean(val_dice_sam_prompt_list)

            train_mean_losses = [mean(x) for x in train_losses]


            logs = ""

            logs += f"Train Dice_sam: {dice_sam_mean}\n"
            logs += f"Train Dice: {dice_mean}\n"
            logs += f"Train Dice_sam_prompt: {dice_sam_prompt_mean}\n"
            logs += f"Train Dice_prompt: {dice_prompt_mean}\n"
            logs += f"Mean train loss: {mean(train_epoch_losses)}\n"

            

            logs += f"val Dice_sam: {val_dice_sam_mean}\n"
            logs += f"val Dice: {val_dice_mean}\n"
            logs += f"val Dice_sam_prompt: {val_dice_sam_prompt_mean}\n"
            logs += f"val Dice_prompt: {val_dice_prompt_mean}\n"

                # plt.plot(val_epochs, val_mean_losses, train_epochs, train_mean_losses)
            if val_dice_prompt_mean > last_best_dice:
                torch.save(
                    conv3d_instance,
                    f"exps/{exp_id}-{user_input}/conv_save.pth",
                )
                print("Model saved")
                last_best_dice = val_dice_prompt_mean


            print(logs)
            log_file.write(logs)
            scheduler.step()
    ## training with k-fold cross validation:


fff = time()
train_model(train_loader, val_loader,test_loader)
log_file.close()

# train and also test the model