import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# def create_prompt(masks, forground=2, background=2):
#     kernel_size = 9
#     kernel = nn.Conv2d(
#         in_channels=1,
#         bias=False,
#         out_channels=1,
#         kernel_size=kernel_size,
#         stride=1,
#         padding=kernel_size // 2,
#     )
#     # print(kernel.weight.shape)
#     kernel.weight = nn.Parameter(
#         torch.zeros(1, 1, kernel_size, kernel_size).to(masks.device),
#         requires_grad=False,
#     )
#     kernel.weight[0, 0] = 1.0
#     eroded_masks = kernel(masks).squeeze(1)//(kernel_size**2)
#     masks = masks.squeeze(1)
#     use_eroded = (eroded_masks.sum(dim=(1, 2), keepdim=True) >= forground).float()
#     new_masks = (eroded_masks * use_eroded) + (masks * (1 - use_eroded))
#     all_points = []
#     all_labels = []
#     for i in range(len(new_masks)):
#         new_background = background
#         points = []
#         labels = []
#         new_mask = new_masks[i]
#         nonzeros = torch.nonzero(new_mask, as_tuple=False)
#         n_nonzero = len(nonzeros)
#         if n_nonzero >= forground:
#             indices = np.random.choice(
#                 np.arange(n_nonzero), size=forground, replace=False
#             ).tolist()
#             # raise ValueError(nonzeros[:, [0, 1]][indices])
#             points.append(nonzeros[:, [1,0]][indices])
#             labels.append(torch.ones(forground))
#         else:
#             if n_nonzero > 0:
#                 points.append(nonzeros)
#                 labels.append(torch.ones(n_nonzero))
#             new_background += forground - n_nonzero
#         # print(points, new_background)
#         zeros = torch.nonzero(1 - masks[i], as_tuple=False)
#         n_zero = len(zeros)
#         indices = np.random.choice(
#             np.arange(n_zero), size=new_background, replace=False
#         ).tolist()
#         points.append(zeros[:, [1, 0]][indices])
#         labels.append(torch.zeros(new_background))
#         points = torch.cat(points, dim=0)
#         labels = torch.cat(labels, dim=0)
#         all_points.append(points)
#         all_labels.append(labels)
#     all_points = torch.stack(all_points, dim=0)
#     all_labels = torch.stack(all_labels, dim=0)
#     return all_points, all_labels 




##########################################################################################1######################################################################################################

# def create_prompt(probabilities, forground=2, background=2):
#     kernel_size = 9
#     kernel = nn.Conv2d(
#         in_channels=1,
#         bias=False,
#         out_channels=1,
#         kernel_size=kernel_size,
#         stride=1,
#         padding=kernel_size // 2,
#     )
#     kernel.weight = nn.Parameter(
#         torch.zeros(1, 1, kernel_size, kernel_size).to(probabilities.device),
#         requires_grad=False,
#     )
#     kernel.weight[0, 0] = 1.0
#     eroded_probs = kernel(probabilities).squeeze(1) / (kernel_size ** 2)
#     probabilities = probabilities.squeeze(1)

#     all_points = []
#     all_labels = []

#     for i in range(len(probabilities)):
#         points = []
#         labels = []

#         prob_mask = probabilities[i]
#         # prob_mask= prob_mask.sigmoid()
#         # raise ValueError(prob_mask1.unique())
#         # print(prob_mask.unique())
        

#         if torch.max(prob_mask) > 0.01:
#             # If probability is greater than 0, select 2 topk points for foreground
#             foreground_indices = torch.topk(prob_mask.view(-1), k=forground, dim=0).indices
#             foreground_points = torch.nonzero(prob_mask > 0, as_tuple=False)
#             n_foreground = len(foreground_points)

#             if n_foreground >= forground:
#                 # Select 2 points from the region where probability is greater than 0
#                 indices_foreground = np.random.choice(np.arange(n_foreground), size=forground, replace=False).tolist()
#                 selected_foreground = foreground_points[indices_foreground]
#                 points.append(selected_foreground[:, [1, 0]])
#                 labels.append(torch.ones(forground))
#             else:
#                 pass
#                 # if n_foreground > 0:
#                 #     points.append(foreground_points[:, [1, 0]])
#                 #     labels.append(torch.ones(n_foreground))


#             # Select 2 background points, one from 0 to -15 and one less than -15
#             background_indices_1 = torch.nonzero((prob_mask < 0.4) & (prob_mask > -0.7), as_tuple=False)
#             background_indices_2 = torch.nonzero((prob_mask <= -2) &(prob_mask>-3), as_tuple=False)

#             # Randomly sample from each set of background points
#             indices_1 = np.random.choice(np.arange(len(background_indices_1)), size=1, replace=False).tolist()
#             indices_2 = np.random.choice(np.arange(len(background_indices_2)), size=1, replace=False).tolist()

#             points.append(background_indices_1[indices_1])
#             points.append(background_indices_2[indices_2])
#             labels.append(torch.zeros(2))
#         else:
#             # If no probability is greater than 0, return 4 background points
#             # print(prob_mask.unique())
#             background_indices_1 = torch.nonzero(prob_mask < 0, as_tuple=False)

#             indices_1 = np.random.choice(np.arange(len(background_indices_1)), size=4, replace=False).tolist()
        
#             points.append(background_indices_1[indices_1])
#             labels.append(torch.zeros(4))

#         points = torch.cat(points, dim=0)
#         labels = torch.cat(labels, dim=0)

#         all_points.append(points)
#         all_labels.append(labels)

#     all_points = torch.stack(all_points, dim=0)
#     all_labels = torch.stack(all_labels, dim=0)
#     # print(all_points, all_labels)

#     return all_points, all_labels

##############################################################################################3##################################################################################################

def distance_to_edge(point, image_shape):
    y, x = point
    height, width = image_shape
    distance_top = y
    distance_bottom = height - y
    distance_left = x
    distance_right = width - x
    return min(distance_top, distance_bottom, distance_left, distance_right)

def create_prompt(probabilities, forground=2, background=2):
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


##############################################################################################2##########################################################################

# def distance_to_edge(point, image_shape):
#     y, x = point
#     height, width = image_shape
#     distance_top = y
#     distance_bottom = height - y
#     distance_left = x
#     distance_right = width - x
#     return min(distance_top, distance_bottom, distance_left, distance_right)

# def create_prompt(probabilities, foreground=2, background=2):
#     kernel_size = 9
#     kernel = nn.Conv2d(
#         in_channels=1,
#         bias=False,
#         out_channels=1,
#         kernel_size=kernel_size,
#         stride=1,
#         padding=kernel_size // 2,
#     )
#     kernel.weight = nn.Parameter(
#         torch.zeros(1, 1, kernel_size, kernel_size).to(probabilities.device),
#         requires_grad=False,
#     )
#     kernel.weight[0, 0] = 1.0
#     eroded_probs = kernel(probabilities).squeeze(1) / (kernel_size ** 2)
#     probabilities = probabilities.squeeze(1)

#     all_points = []
#     all_labels = []

#     for i in range(len(probabilities)):
#         points = []
#         labels = []

#         prob_mask = probabilities[i]

#         if torch.max(prob_mask) > 0.01:
#             foreground_indices = torch.topk(prob_mask.view(-1), k=foreground, dim=0).indices
#             foreground_points = torch.nonzero(prob_mask > 0, as_tuple=False)
#             n_foreground = len(foreground_points)
#             if n_foreground >= foreground:
#                 # Get the index of the point with the highest probability
#                 top_prob_idx = torch.topk(prob_mask.view(-1), k=1).indices[0]
#                 # Convert the flat index to 2D coordinates
#                 top_prob_point = np.unravel_index(top_prob_idx.item(), prob_mask.shape)
#                 top_prob_point = torch.tensor(top_prob_point, device=probabilities.device)  # Move to the same device

#                 # Add the point with the highest probability to the points list
#                 points.append(torch.tensor([top_prob_point[1], top_prob_point[0]], device=probabilities.device).unsqueeze(0))
#                 labels.append(torch.ones(1, device=probabilities.device))

#                 # Exclude the top probability point when finding the point closest to the edge
#                 remaining_foreground_points = foreground_points[(foreground_points != top_prob_point.unsqueeze(0)).all(dim=1)]
#                 if remaining_foreground_points.numel() > 0:
#                     distances = [distance_to_edge(point.cpu().numpy(), prob_mask.shape) for point in remaining_foreground_points]
#                     edge_point_idx = np.argmin(distances)
#                     edge_point = remaining_foreground_points[edge_point_idx]

#                     # Add the edge point to the points list
#                     points.append(edge_point[[1, 0]].unsqueeze(0))
#                     labels.append(torch.ones(1, device=probabilities.device))
#                     # raise ValueError(points , labels)
#             else:
#                 if n_foreground > 0:
#                     points.append(foreground_points[:, [1, 0]])
#                     labels.append(torch.ones(n_foreground))



#             # Select 2 background points, one from 0 to -15 and one less than -15
#             background_indices_1 = torch.nonzero((prob_mask < 0) & (prob_mask > -15), as_tuple=False)
#             background_indices_2 = torch.nonzero(prob_mask < -15, as_tuple=False)

#             # Randomly sample from each set of background points
#             indices_1 = np.random.choice(np.arange(len(background_indices_1)), size=1, replace=False).tolist()
#             indices_2 = np.random.choice(np.arange(len(background_indices_2)), size=1, replace=False).tolist()

#             points.append(background_indices_1[indices_1])
#             points.append(background_indices_2[indices_2])
#             labels.append(torch.zeros(2))
#         else:
#             # If no probability is greater than 0, return 4 background points
#             # print(prob_mask.unique())
#             background_indices_1 = torch.nonzero(prob_mask < 0, as_tuple=False)

#             indices_1 = np.random.choice(np.arange(len(background_indices_1)), size=4, replace=False).tolist()
        
#             points.append(background_indices_1[indices_1])
#             labels.append(torch.zeros(4))
            
#         labels = [label.to(probabilities.device) for label in labels]
#         points = torch.cat(points, dim=0)

#         all_points.append(points)
#         all_labels.append(torch.cat(labels, dim=0))


#     all_points = torch.stack(all_points, dim=0)
#     all_labels = torch.stack(all_labels, dim=0)
#     # print(all_points, all_labels)
#     # raise ValueError(all_points , all_labels )

#     return all_points, all_labels
##########################################################################################

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

#############################versionGroundTure################################################
# def main_prompt_for_ground_true(probabilities):
#     probabilities = probabilities
#     # raise ValueError(probabilities)

#     # Thresholding function
#     def threshold(tensor, thresh):
#         return (tensor > thresh).float()

#     # Morphological operations
#     def morphological_op(tensor, operation, kernel_size):
#         kernel = torch.ones(1, 1, kernel_size[0], kernel_size[1]).to(tensor.device)
#         if kernel_size[0] % 2 == 0:  
#             padding = [(k - 1) // 2 for k in kernel_size]
#             extra_pad = [0, 2, 0, 2]  
#         else:
#             padding = [(k - 1) // 2 for k in kernel_size]
#             extra_pad = [0, 0, 0, 0]  

#         if operation == 'erode':
#             tensor = F.conv2d(F.pad(tensor, extra_pad), kernel, padding=padding).clamp(max=1)
#         elif operation == 'dilate':
#             tensor = F.max_pool2d(F.pad(tensor, extra_pad), kernel_size, stride=1, padding=padding).clamp(max=1)

#         if kernel_size[0] % 2 == 0:  
#             tensor = tensor[:, :, :tensor.shape[2] - 1, :tensor.shape[3] - 1]

#         return tensor.squeeze(1)

#     # Foreground prompts
#     th_O = threshold(probabilities, 0.5)
#     M_f = morphological_op(morphological_op(th_O, 'erode', (10, 10)), 'dilate', (5, 5))
#     foreground_indices = torch.nonzero(M_f.squeeze(0), as_tuple=False)
#     n_for = 2 if len(foreground_indices) >= 2 else len(foreground_indices)
#     n_back = 4 - n_for
#     # Background prompts
#     M_b1 = 1 - morphological_op(threshold(probabilities, 0.5), 'dilate', (10, 10))
#     M_b2 = 1 - threshold(probabilities, 0.4)
#     M_b2 = M_b2.squeeze(1)

#     M_b = M_b1 * M_b2
#     M_b = M_b.squeeze(0)
#     background_indices = torch.nonzero(M_b, as_tuple=False)

#     if n_for > 0:
#         indices = torch.concat([foreground_indices[np.random.choice(np.arange(len(foreground_indices)), size=n_for)],
#                                 background_indices[np.random.choice(np.arange(len(background_indices)), size=n_back)]
#                                 ])
#         values = torch.tensor([1] * n_for + [0] * n_back)
#     else:
#         indices = background_indices[np.random.choice(np.arange(len(background_indices)), size=4)]
#         values = torch.tensor([0] * 4)
#     # raise ValueError(indices, values)
#     # raise ValueError(indices.shape)
#     # raise ValueError(indices)
    
#     modified_indices = indices[:, 1:]  

#     return indices.unsqueeze(0), values.unsqueeze(0)


#############################################################################################

