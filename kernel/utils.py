import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def create_prompt_simple(masks, forground=2, background=2):
    kernel_size = 9
    kernel = nn.Conv2d(
        in_channels=1,
        bias=False,
        out_channels=1,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    # print(kernel.weight.shape)
    kernel.weight = nn.Parameter(
        torch.zeros(1, 1, kernel_size, kernel_size).to(masks.device),
        requires_grad=False,
    )
    kernel.weight[0, 0] = 1.0
    eroded_masks = kernel(masks).squeeze(1)//(kernel_size**2)
    masks = masks.squeeze(1)
    use_eroded = (eroded_masks.sum(dim=(1, 2), keepdim=True) >= forground).float()
    new_masks = (eroded_masks * use_eroded) + (masks * (1 - use_eroded))
    all_points = []
    all_labels = []
    for i in range(len(new_masks)):
        new_background = background
        points = []
        labels = []
        new_mask = new_masks[i]
        nonzeros = torch.nonzero(new_mask, as_tuple=False)
        n_nonzero = len(nonzeros)
        if n_nonzero >= forground:
            indices = np.random.choice(
                np.arange(n_nonzero), size=forground, replace=False
            ).tolist()
            # raise ValueError(nonzeros[:, [0, 1]][indices])
            points.append(nonzeros[:, [1,0]][indices])
            labels.append(torch.ones(forground))
        else:
            if n_nonzero > 0:
                points.append(nonzeros)
                labels.append(torch.ones(n_nonzero))
            new_background += forground - n_nonzero
        # print(points, new_background)
        zeros = torch.nonzero(1 - masks[i], as_tuple=False)
        n_zero = len(zeros)
        indices = np.random.choice(
            np.arange(n_zero), size=new_background, replace=False
        ).tolist()
        points.append(zeros[:, [1, 0]][indices])
        labels.append(torch.zeros(new_background))
        points = torch.cat(points, dim=0)
        labels = torch.cat(labels, dim=0)
        all_points.append(points)
        all_labels.append(labels)
    all_points = torch.stack(all_points, dim=0)
    all_labels = torch.stack(all_labels, dim=0)
    return all_points, all_labels 




#
device = "cuda:0"
def create_prompt_main(probabilities):
    probabilities = probabilities.sigmoid()

    # Thresholding function
    def threshold(tensor, thresh):
        return (tensor > thresh).float()

    # Morphological operations
    def morphological_op(tensor, operation, kernel_size):
        kernel = torch.ones(1, 1, kernel_size[0], kernel_size[1]).to(tensor.device)
        if kernel_size[0] % 2 == 0:  
            padding = [(k - 1) // 2 for k in kernel_size]
            extra_pad = [0, 2, 0, 2]  
        else:
            padding = [(k - 1) // 2 for k in kernel_size]
            extra_pad = [0, 0, 0, 0]  

        if operation == 'erode':
            tensor = F.conv2d(F.pad(tensor, extra_pad), kernel, padding=padding).clamp(max=1)
        elif operation == 'dilate':
            tensor = F.max_pool2d(F.pad(tensor, extra_pad), kernel_size, stride=1, padding=padding).clamp(max=1)

        if kernel_size[0] % 2 == 0:  
            tensor = tensor[:, :, :tensor.shape[2] - 1, :tensor.shape[3] - 1]

        return tensor.squeeze(1)

    # Foreground prompts
    th_O = threshold(probabilities, 0.5)
    M_f = morphological_op(morphological_op(th_O, 'erode', (10, 10)), 'dilate', (5, 5))
    foreground_indices = torch.nonzero(M_f.squeeze(0), as_tuple=False)
    n_for = 2 if len(foreground_indices) >= 2 else len(foreground_indices)
    n_back = 4 - n_for
    # Background prompts
    M_b1 = 1 - morphological_op(threshold(probabilities, 0.5), 'dilate', (10, 10))
    M_b2 = 1 - threshold(probabilities, 0.4)
    M_b2 = M_b2.squeeze(1)

    M_b = M_b1 * M_b2
    M_b = M_b.squeeze(0)
    background_indices = torch.nonzero(M_b, as_tuple=False)

    if n_for > 0:
        indices = torch.concat([foreground_indices[np.random.choice(np.arange(len(foreground_indices)), size=n_for)],
                                background_indices[np.random.choice(np.arange(len(background_indices)), size=n_back)]
                                ])
        values = torch.tensor([1] * n_for + [0] * n_back)
    else:
        indices = background_indices[np.random.choice(np.arange(len(background_indices)), size=4)]
        values = torch.tensor([0] * 4)
    # raise ValueError(indices, values)
    return indices.unsqueeze(0), values.unsqueeze(0)



