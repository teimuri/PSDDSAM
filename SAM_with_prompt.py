debug = 0
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
from einops import rearrange
import random
from tqdm import tqdm
from time import sleep
from data import *
from time import time
from PIL import Image
from sklearn.model_selection import KFold
from shutil import copyfile
# import monai
from tqdm import tqdm
from utils import create_prompt_armin,create_prompt_armin_for_ground_true
from torch.autograd import Variable

# import wandb_handler


def save_img(img, dir):
    img = img.clone().cpu().numpy() + 100
    if len(img.shape) == 3:
        img = rearrange(img, "c h w -> h w c")
        img_min = np.amin(img, axis=(0, 1), keepdims=True)
        img = img - img_min

        img_max = np.amax(img, axis=(0, 1), keepdims=True)
        img = (img / img_max * 255).astype(np.uint8)
        grey_img = Image.fromarray(img[:, :, 0])
        img = Image.fromarray(img)

    else:
        img_min = img.min()
        img = img - img_min
        img_max = img.max()
        if img_max != 0:
            img = img / img_max * 255
        img = Image.fromarray(img).convert("L")

    img.save(dir)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def dice_loss(self, logits, gt, eps=1):
        # Convert logits to probabilities
        # Flatten the tensors
        # probs = probs.view(-1)
        # gt = gt.view(-1)

        probs = torch.sigmoid(logits)

        # Compute Dice coefficient
        intersection = (probs * gt).sum()

        dice_coeff = (2.0 * intersection + eps) / (probs.sum() + gt.sum() + eps)

        # Compute Dice Los[s
        loss = 1 - dice_coeff
        return loss

    def focal_loss(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        # pred=pred.reshape(-1,1)
        # mask = mask.reshape(-1,1)
        # assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p**self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss

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


class loss_fn(torch.nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, epsilon=1e-5):
        super(loss_fn, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def tversky_loss(self, y_pred, y_true, alpha=0.8, beta=0.2, smooth=1e-2):
        y_pred = torch.sigmoid(y_pred)
        # raise ValueError(y_pred)
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        tversky_index = (true_pos + smooth) / (
            true_pos + alpha * false_neg + beta * false_pos + smooth
        )
        return 1 - tversky_index

    def focal_tversky(self, y_pred, y_true, gamma=0.75):
        pt_1 = self.tversky_loss(y_pred, y_true)
        return torch.pow((1 - pt_1), gamma)
    def dice_loss(self, logits, gt, eps=1):
        # Convert logits to probabilities
        # Flatten the tensorsx
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        gt = gt.view(-1)

        # Compute Dice coefficient
        intersection = (probs * gt).sum()

        dice_coeff = (2.0 * intersection + eps) / (probs.sum() + gt.sum() + eps)

        # Compute Dice Los[s
        loss = 1 - dice_coeff
        return loss

    def focal_loss(self, logits, gt, gamma=2):
        logits = logits.reshape(-1, 1)
        gt = gt.reshape(-1, 1)
        logits = torch.cat((1 - logits, logits), dim=1)

        probs = torch.sigmoid(logits)
        pt = probs.gather(1, gt.long())

        modulating_factor = (1 - pt) ** gamma
        # pt_false= pt<=0.5
        # modulating_factor[pt_false] *= 2
        focal_loss = -modulating_factor * torch.log(pt + 1e-12)

        # Compute the mean focal loss
        loss = focal_loss.mean()
        return loss  # Store as a Python number to save memory

    def forward(self, logits, target):
        logits = logits.squeeze(1)
        target = target.squeeze(1)
        # Dice Loss
        # prob = F.softmax(logits, dim=1)[:, 1, ...]

        dice_loss = self.dice_loss(logits, target)
        tversky_loss = self.tversky_loss(logits, target)

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


def dice_coefficient(pred, target):
    smooth = 1  # Smoothing constant to avoid division by zero
    dice = 0
    pred_index = pred
    target_index = target
    intersection = (pred_index * target_index).sum()
    union = pred_index.sum() + target_index.sum()
    dice += (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


accumaltive_batch_size = 512
batch_size = 1
num_workers = 4
slice_per_image = 1
num_epochs = 80
sample_size = 2000
# image_size=sam_model.image_encoder.img_size
image_size = 1024
exp_id = 0
found = 0
if debug:
    user_input = "debug"
else:
    user_input = input("Related changes: ")
while found == 0:
    try:
        os.makedirs(f"exps/{exp_id}-{user_input}")
        found = 1
    except:
        exp_id = exp_id + 1
copyfile(os.path.realpath(__file__), f"exps/{exp_id}-{user_input}/code.py")


layer_n = 4
L = layer_n
a = np.full(L, layer_n)
params = {"M": 255, "a": a, "p": 0.35}


model_type = "vit_h"
checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
device = "cuda:0"


from segment_anything import SamPredictor, sam_model_registry


# //////////////////
class panc_sam(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        # self.sam = torch.load(
        #     "/mnt/new_drive/PanCanAid/PanCanAid-segmentation/exps/correct_prompt/sam_tuned_save.pth"
        # ).sam


    def forward(self, batched_input):
        # with torch.no_grad():
        # raise ValueError(10)
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(input_images).detach()
        
        outputs = []
        
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"].unsqueeze(0), image_record["point_labels"].unsqueeze(0))
                # raise ValueError(points)
            else:
                raise ValueError('what the f?')
                points = None
            # raise ValueError(image_record["point_coords"].shape)
            with torch.no_grad():
                 sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                     points=points,
                     boxes=image_record.get("boxes", None),
                     masks=image_record.get("mask_inputs", None),
                 )

            
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe().detach(),
                sparse_prompt_embeddings=sparse_embeddings.detach(),
                dense_prompt_embeddings=dense_embeddings.detach(),
                multimask_output=False,
            )
            outputs.append(
                {
                    "low_res_logits": low_res_masks,
                }
            )
        low_res_masks = torch.stack([x["low_res_logits"] for x in outputs], dim=0)

        return low_res_masks.squeeze(1)


# ///////////////

augmentation = A.Compose(
    [
        A.Rotate(limit=90, p=0.5),
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
        A.RandomScale(scale_limit=0.1, p=0.5),

    ]
)
panc_sam_instance = panc_sam()

panc_sam_instance.to(device)
panc_sam_instance.train()

train_dataset = PanDataset(
    [
     "/media/external_2T/malekahmadi/PanCanAid/Data/NIH_PNG/train/images"],
    [
     "/media/external_2T/malekahmadi/PanCanAid/Data/NIH_PNG/train/labels"],
    # ["/mnt/new_drive/PanCanAid/Data/NIH_PNG/train/images"],
    # ["/mnt/new_drive/PanCanAid/Data/NIH_PNG/train/labels"],
    [["NIH_PNG",1]],
    
    image_size,
    
    slice_per_image=slice_per_image,
    train=True,
    augmentation=augmentation,
)
test_dataset = PanDataset(
    [
     "/media/external_2T/malekahmadi/PanCanAid/Data/NIH_PNG/test/images"],
    [
     "/media/external_2T/malekahmadi/PanCanAid/Data/NIH_PNG/test/labels"],
        
    [["NIH_PNG",1]],

    image_size,
    
    slice_per_image=slice_per_image,
    train=False,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=train_dataset.collate_fn,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=test_dataset.collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)


# Set up the optimizer, hyperparameter tuning will improve performance here
lr = 1e-4
max_lr = 5e-5
wd = 5e-4


optimizer = torch.optim.Adam(
    # parameters,
    list(panc_sam_instance.sam.mask_decoder.parameters()),
    # list(panc_sam_instance.mask_decoder.parameters()),
    lr=lr,
    weight_decay=wd,
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=num_epochs,
    steps_per_epoch=sample_size // (accumaltive_batch_size // batch_size),
)

from statistics import mean

from tqdm import tqdm
from torch.nn.functional import threshold, normalize

loss_function = loss_fn(alpha=0.5, gamma=2.0)
loss_function.to(device)

from time import time
import time as s_time

log_file = open(f"exps/{exp_id}-{user_input}/log.txt", "a")


def process_model(data_loader, train=0, save_output=0):
    epoch_losses = []

    index = 0
    results = torch.zeros((2, 0, 256, 256))
    total_dice = 0.0
    num_samples = 0

    counterb = 0
    for image, label in tqdm(data_loader, total=sample_size):
        s_time.sleep(0.6)
        counterb += 1

        index += 1
        image = image.to(device)
        label = label.to(device).float()

        input_size = (1024, 1024)

        box = torch.tensor([[200, 200, 750, 800]]).to(device)
        points, point_labels = create_prompt_armin_for_ground_true(label)
        # raise ValueError(points)
        batched_input = []
        for ibatch in range(batch_size):
            
            batched_input.append(
                {
                    "image": image[ibatch],
                    "point_coords": points[ibatch],
                    "point_labels": point_labels[ibatch],
                    "original_size": (1024, 1024)
                    # 'original_size': image1.shape[:2]
                },
            )
            # raise ValueError(batched_input)

        low_res_masks = panc_sam_instance(batched_input)
        low_res_label = F.interpolate(label, low_res_masks.shape[-2:])    
        binary_mask = normalize(threshold(low_res_masks, 0.0,0))
        loss = loss_function(low_res_masks, low_res_label)
            
        loss /= (accumaltive_batch_size / batch_size)
        opened_binary_mask = torch.zeros_like(binary_mask).cpu()

        for j, mask in enumerate(binary_mask[:, 0]):
            numpy_mask = mask.detach().cpu().numpy().astype(np.uint8)

            opened_binary_mask[j][0] = torch.from_numpy(numpy_mask)

        dice = dice_coefficient(
            opened_binary_mask.numpy(), low_res_label.cpu().detach().numpy()
        )
        # print(dice)
        total_dice += dice
        num_samples += 1
        average_dice = total_dice / num_samples
        log_file.write(str(average_dice) + "\n")
        log_file.flush()
        if train:
            loss.backward()

            if index % (accumaltive_batch_size / batch_size) == 0:
                # print(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                index = 0

        else:
            result = torch.cat(
                (
                    low_res_masks[0].detach().cpu().reshape(1, 1, 256, 256),
                    opened_binary_mask[0].reshape(1, 1, 256, 256),
                ),
                dim=0,
            )
            results = torch.cat((results, result), dim=1)
        if index % (accumaltive_batch_size / batch_size) == 0:
            epoch_losses.append(loss.item())
        if counterb == sample_size and train:
            break
        elif counterb == sample_size // 5 and not train:
            break

    return epoch_losses, results, average_dice


def train_model(train_loader, test_loader, K_fold=False, N_fold=7, epoch_num_start=7):
    print("Train model started.")

    train_losses = []
    train_epochs = []
    test_losses = []
    test_epochs = []
    dice = []
    dice_test = []
    results = []
    if debug==0:
        index = 0
        # for image, label in tqdm(test_loader):
            
        #     points, point_labels = create_prompt_armin(label)
        #     points = torch.cat((point_labels[0].unsqueeze(1),points[0]),dim=1).long()

        #     # for point in points:
        #     #     if point[0]:
        #     #         image[0][point[0],point[2]-5:point[2]+5,point[1]-5:point[1]+5]=5
        #     #     else:
        #     #         image[0][point[0],point[1]-5:point[1]+5,point[2]-5:point[2]+5]=5

            
        #     # if index < 100:
        #     #     if not os.path.exists(f"ims/batch_{index}"):
        #     #         os.mkdir(f"ims/batch_{index}")

        #     #     save_img(
        #     #         image[0],
        #     #         f"ims/batch_{index}/img_0.png",
        #     #     )
        #     #     save_img(0.2 * image[0][0] + label[0][0], f"ims/batch_{index}/gt_0.png")
            

        #     index += 1
        #     if index == 100:
        #         break



    ## training with k-fold cross validation:
    last_best_dice = 0
    for epoch in range(num_epochs):
        if epoch > epoch_num_start:
            kf = KFold(n_splits=N_fold, shuffle=True)
            for i, (train_index, test_index) in enumerate(kf.split(train_loader)):
                print(
                    f"=====================EPOCH: {epoch} fold: {i}====================="
                )
                print("Training:")
                x_train, x_test = (
                    train_loader[train_index],
                    train_loader[test_index],
                )

                train_epoch_losses, epoch_results, average_dice = process_model(
                    x_train, train=1
                )

                dice.append(average_dice)
                train_losses.append(train_epoch_losses)
                if (average_dice) > 0.6:
                    print("Testing:")
                    (
                        test_epoch_losses,
                        epoch_results,
                        average_dice_test,
                    ) = process_model(x_test)

                    test_losses.append(test_epoch_losses)
                    for i in tqdm(range(len(epoch_results[0]))):
                        if not os.path.exists(f"ims/batch_{i}"):
                            os.mkdir(f"ims/batch_{i}")

                        save_img(
                            epoch_results[0, i].clone(),
                            f"ims/batch_{i}/prob_epoch_{epoch}.png",
                        )
                        save_img(
                            epoch_results[1, i].clone(),
                            f"ims/batch_{i}/pred_epoch_{epoch}.png",
                        )
                train_mean_losses = [mean(x) for x in train_losses]
                test_mean_losses = [mean(x) for x in test_losses]

                np.save("train_losses.npy", train_mean_losses)
                np.save("test_losses.npy", test_mean_losses)

                print(f"Train Dice: {average_dice}")
                print(f"Mean train loss: {mean(train_epoch_losses)}")

                try:
                    dice_test.append(average_dice_test)
                    print(f"Test Dice : {average_dice_test}")
                    print(f"Mean test loss: {mean(test_epoch_losses)}")

                    results.append(epoch_results)
                    test_epochs.append(epoch)
                    train_epochs.append(epoch)
                    plt.plot(
                        test_epochs,
                        test_mean_losses,
                        train_epochs,
                        train_mean_losses,
                    )
                    if average_dice_test > last_best_dice:
                        torch.save(
                            panc_sam_instance,
                            f"exps/{exp_id}-{user_input}/sam_tuned_save.pth",
                        )

                        last_best_dice = average_dice_test
                    del epoch_results
                    del average_dice_test
                except:
                    train_epochs.append(epoch)
                    plt.plot(train_epochs, train_mean_losses)
                    print(
                        f"=================End of EPOCH: {epoch} Fold :{i}==================\n"
                    )

            plt.yscale("log")
            plt.title("Mean epoch loss")
            plt.xlabel("Epoch Number")
            plt.ylabel("Loss")
            plt.savefig("result")

        else:
            print(f"=====================EPOCH: {epoch}=====================")
            last_best_dice = 0
            print("Training:")
            train_epoch_losses, epoch_results, average_dice = process_model(
                train_loader, train=1
            )

            dice.append(average_dice)
            train_losses.append(train_epoch_losses)
            if (average_dice) > 0.6:
                print("Testing:")
                test_epoch_losses, epoch_results, average_dice_test = process_model(
                    test_loader
                )

                test_losses.append(test_epoch_losses)
                # for i in tqdm(range(len(epoch_results[0]))):
                #     if not os.path.exists(f"ims/batch_{i}"):
                #         os.mkdir(f"ims/batch_{i}")

                #     save_img(
                #         epoch_results[0, i].clone(),
                #         f"ims/batch_{i}/prob_epoch_{epoch}.png",
                #     )
                #     save_img(
                #         epoch_results[1, i].clone(),
                #         f"ims/batch_{i}/pred_epoch_{epoch}.png",
                #     )

            train_mean_losses = [mean(x) for x in train_losses]
            test_mean_losses = [mean(x) for x in test_losses]

            np.save("train_losses.npy", train_mean_losses)
            np.save("test_losses.npy", test_mean_losses)

            print(f"Train Dice: {average_dice}")
            print(f"Mean train loss: {mean(train_epoch_losses)}")

            try:
                dice_test.append(average_dice_test)
                print(f"Test Dice : {average_dice_test}")
                print(f"Mean test loss: {mean(test_epoch_losses)}")

                results.append(epoch_results)
                test_epochs.append(epoch)
                train_epochs.append(epoch)
                plt.plot(
                    test_epochs, test_mean_losses, train_epochs, train_mean_losses
                )
                if average_dice_test > last_best_dice:
                    torch.save(
                        panc_sam_instance,
                        f"exps/{exp_id}-{user_input}/sam_tuned_save.pth",
                    )

                    last_best_dice = average_dice_test
                del epoch_results
                del average_dice_test
            except:
                train_epochs.append(epoch)
                plt.plot(train_epochs, train_mean_losses)
                print(f"=================End of EPOCH: {epoch}==================\n")

            plt.yscale("log")
            plt.title("Mean epoch loss")
            plt.xlabel("Epoch Number")
            plt.ylabel("Loss")
            plt.savefig("result")

    return train_losses, test_losses, results


train_losses, test_losses, results = train_model(train_loader, test_loader)
log_file.close()

# train and also test the model
