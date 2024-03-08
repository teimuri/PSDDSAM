import torch
from torch import nn
import numpy as np



class loss_fn(torch.nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, epsilon=1e-5):
        super(loss_fn, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon


    def focal_tversky(self, y_pred, y_true, gamma=0.75):
        pt_1 = self.tversky_loss(y_pred, y_true)
        return torch.pow((1 - pt_1), gamma)
    def dice_loss(self, probs, gt, eps=1):
        
        intersection = (probs * gt).sum(dim=(-2,-1))
        dice_coeff = (2.0 * intersection + eps) / (probs.sum(dim=(-2,-1)) + gt.sum(dim=(-2,-1)) + eps)
        loss = 1 - dice_coeff.mean()
        return loss

    def focal_loss(self, probs, gt, gamma=4):
        probs = probs.reshape(-1, 1)
        gt = gt.reshape(-1, 1)
        probs = torch.cat((1 - probs, probs), dim=1)

        pt = probs.gather(1, gt.long())
        modulating_factor = (1 - pt) ** gamma
        # modulating_factor = (3**(10*((1-pt)-0.5)))*(1 - pt) ** gamma
        modulating_factor[pt>0.55] = 0.1*modulating_factor[pt>0.55] 

        focal_loss = -modulating_factor * torch.log(pt + 1e-12)
        
        # Compute the mean focal loss
        loss = focal_loss.mean()
        return loss  # Store as a Python number to save memory

    def forward(self, probs, target):
        
        self.gamma=8
        dice_loss = self.dice_loss(probs, target)
        # tversky_loss = self.tversky_loss(logits, target)

        # Focal Loss
        focal_loss = self.focal_loss(probs, target,self.gamma)
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
    # raise ValueError( binary_mask.shape,gt.shape)
    intersection = (binary_mask * gt).sum(dim=(-2,-1))
    dice_scores = (2.0 * intersection + eps) / (binary_mask.sum(dim=(-2,-1)) + gt.sum(dim=(-2,-1)) + eps)
    # raise ValueError(intersection.shape , binary_mask.shape,gt.shape)
    
    return dice_scores.mean()

def calculate_accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

def calculate_sensitivity(pred, target): 
    smooth = 1
    # Also known as recall
    true_positive = ((pred == 1) & (target == 1)).sum().item()
    false_negative = ((pred == 0) & (target == 1)).sum().item()
    return (true_positive + smooth) / ((true_positive + false_negative) + smooth)

def calculate_specificity(pred, target):
    smooth = 1
    true_negative = ((pred == 0) & (target == 0)).sum().item()
    false_positive = ((pred == 1) & (target == 0)).sum().item()
    return (true_negative + smooth) / ((true_negative + false_positive ) + smooth)