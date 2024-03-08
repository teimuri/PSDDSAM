import torch
import torch.nn as nn
import numpy as np


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



def distance_to_edge(point, image_shape):
    y, x = point
    height, width = image_shape
    distance_top = y
    distance_bottom = height - y
    distance_left = x
    distance_right = width - x
    return min(distance_top, distance_bottom, distance_left, distance_right)

def create_prompt(probabilities, foreground=2, background=2):
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
            foreground_indices = torch.topk(prob_mask.view(-1), k=foreground, dim=0).indices
            foreground_points = torch.nonzero(prob_mask > 0, as_tuple=False)
            n_foreground = len(foreground_points)
            if n_foreground >= foreground:
                # Get the index of the point with the highest probability
                top_prob_idx = torch.topk(prob_mask.view(-1), k=1).indices[0]
                # Convert the flat index to 2D coordinates
                top_prob_point = np.unravel_index(top_prob_idx.item(), prob_mask.shape)
                top_prob_point = torch.tensor(top_prob_point, device=probabilities.device)  # Move to the same device

                # Add the point with the highest probability to the points list
                points.append(torch.tensor([top_prob_point[1], top_prob_point[0]], device=probabilities.device).unsqueeze(0))
                labels.append(torch.ones(1, device=probabilities.device))

                # Exclude the top probability point when finding the point closest to the edge
                remaining_foreground_points = foreground_points[(foreground_points != top_prob_point.unsqueeze(0)).all(dim=1)]
                if remaining_foreground_points.numel() > 0:
                    distances = [distance_to_edge(point.cpu().numpy(), prob_mask.shape) for point in remaining_foreground_points]
                    edge_point_idx = np.argmin(distances)
                    edge_point = remaining_foreground_points[edge_point_idx]

                    # Add the edge point to the points list
                    points.append(edge_point[[1, 0]].unsqueeze(0))
                    labels.append(torch.ones(1, device=probabilities.device))
                    # raise ValueError(points , labels)
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
            
        labels = [label.to(probabilities.device) for label in labels]
        points = torch.cat(points, dim=0)

        all_points.append(points)
        all_labels.append(torch.cat(labels, dim=0))


    all_points = torch.stack(all_points, dim=0)
    all_labels = torch.stack(all_labels, dim=0)
    # print(all_points, all_labels)

    return all_points, all_labels




