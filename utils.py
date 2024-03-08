import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def distance_to_edge(point, image_shape):
    y, x = point
    height, width = image_shape
    distance_top = y
    distance_bottom = height - y
    distance_left = x
    distance_right = width - x
    return min(distance_top, distance_bottom, distance_left, distance_right)

def sample_prompt(probabilities, forground=2, background=2):
    kernel_size = 9
    kernel = nn.Conv2d(
        in_channels=1,
        bias=False,
        out_channels=1,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    kernel.weight = nn.Parameter(
        torch.zeros(1, 1, kernel_size, kernel_size).to(probabilities.device),
        requires_grad=False,
    )
    kernel.weight[0, 0] = 1.0
    eroded_probs = kernel(probabilities).squeeze(1) / (kernel_size ** 2)
    probabilities = probabilities.squeeze(1)

    all_points = []
    all_labels = []

    for i in range(len(probabilities)):
        points = []
        labels = []

        prob_mask = probabilities[i]

        if torch.max(prob_mask) > 0.01:
            foreground_indices = torch.topk(prob_mask.view(-1), k=forground, dim=0).indices
            foreground_points = torch.nonzero(prob_mask > 0, as_tuple=False)
            n_foreground = len(foreground_points)

            if n_foreground >= forground:
                # Calculate distance to edge for each point
                distances = [distance_to_edge(point.cpu().numpy(), prob_mask.shape) for point in foreground_points]

                # Find the point with minimum distance to edge
                edge_point_idx = np.argmin(distances)
                edge_point = foreground_points[edge_point_idx]

                # Append the point closest to the edge and another random point
                points.append(edge_point[[1, 0]].unsqueeze(0))
                indices_foreground = np.random.choice(np.arange(n_foreground), size=forground-1, replace=False).tolist()
                selected_foreground = foreground_points[indices_foreground]
                points.append(selected_foreground[:, [1, 0]])
                labels.append(torch.ones(forground))
            else:
                if n_foreground > 0:
                    points.append(foreground_points[:, [1, 0]])
                    labels.append(torch.ones(n_foreground))



            # Select 2 background points, one from 0 to -15 and one less than -15
            background_indices_1 = torch.nonzero((prob_mask < 0) & (prob_mask > -15), as_tuple=False)
            background_indices_2 = torch.nonzero(prob_mask < -15, as_tuple=False)

            # Randomly sample from each set of background points
            indices_1 = np.random.choice(np.arange(len(background_indices_1)), size=1, replace=False).tolist()
            indices_2 = np.random.choice(np.arange(len(background_indices_2)), size=1, replace=False).tolist()

            points.append(background_indices_1[indices_1])
            points.append(background_indices_2[indices_2])
            labels.append(torch.zeros(2))
        else:
            # If no probability is greater than 0, return 4 background points
            # print(prob_mask.unique())
            background_indices_1 = torch.nonzero(prob_mask < 0, as_tuple=False)

            indices_1 = np.random.choice(np.arange(len(background_indices_1)), size=4, replace=False).tolist()
        
            points.append(background_indices_1[indices_1])
            labels.append(torch.zeros(4))

        points = torch.cat(points, dim=0)
        labels = torch.cat(labels, dim=0)

        all_points.append(points)
        all_labels.append(labels)

    all_points = torch.stack(all_points, dim=0)
    all_labels = torch.stack(all_labels, dim=0)
    # print(all_points, all_labels)

    return all_points, all_labels



device = "cuda:0"
def main_prompt(probabilities):
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


