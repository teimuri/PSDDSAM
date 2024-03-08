import torch
from torch import nn
from torch.nn import functional as F
from utils import create_prompt_armin

device = 'cuda:0'


from segment_anything import SamPredictor, sam_model_registry


class panc_sam(nn.Module):
    
    def forward(self, batched_input, device):
        box = torch.tensor([[200, 200, 750, 800]]).to(device)
        outputs = []
        outputs_prompt = []
        for image_record in batched_input:
            image_embeddings = image_record["image_embedd"].to(device)
            if "point_coords" in image_record:
                point_coords = image_record["point_coords"].to(device)
                point_labels = image_record["point_labels"].to(device)
                points = (point_coords.unsqueeze(0), point_labels.unsqueeze(0))

            else:
                raise ValueError("what the f?")
            # input_images = torch.stack([x["image"] for x in batched_input], dim=0)

            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=box,
                    masks=None,
                )
                sparse_embeddings = sparse_embeddings
                dense_embeddings = dense_embeddings
                # raise ValueError(image_embeddings.shape)
                #####################################################

                low_res_masks, _ = self.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.prompt_encoder.get_dense_pe().detach(),
                    sparse_prompt_embeddings=sparse_embeddings.detach(),
                    dense_prompt_embeddings=dense_embeddings.detach(),
                    multimask_output=False,
                )

                outputs.append(low_res_masks)

                # points, point_labels = create_prompt((low_res_masks > 0).float())
                # points, point_labels = create_prompt(low_res_masks)
                points, point_labels = create_prompt_armin(low_res_masks)


                points = points * 4
                points = (points, point_labels)

                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = self.prompt_encoder2(
                        points=points,
                        boxes=None,
                        masks=None,
                    )

                low_res_masks, _ = self.mask_decoder2(
                    image_embeddings=image_embeddings,
                    image_pe=self.prompt_encoder2.get_dense_pe().detach(),
                    sparse_prompt_embeddings=sparse_embeddings.detach(),
                    dense_prompt_embeddings=dense_embeddings.detach(),
                    multimask_output=False,
                )

                outputs_prompt.append(low_res_masks)

        low_res_masks_promtp = torch.cat(outputs_prompt, dim=1)
        low_res_masks = torch.cat(outputs, dim=1)

        return low_res_masks, low_res_masks_promtp





def double_conv_3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        nn.ReLU(inplace=True),
    )

#Was not used
class UNet3D(nn.Module):

    def __init__(self):
        super(UNet3D, self).__init__()

        self.dconv_down1 = double_conv_3d(1, 32)
        self.dconv_down2 = double_conv_3d(32, 64)
        self.dconv_down3 = double_conv_3d(64, 96)

        self.maxpool = nn.MaxPool3d((1, 2, 2))
        self.upsample = nn.Upsample(
            scale_factor=(1, 2, 2), mode="trilinear", align_corners=True
        )

        self.dconv_up2 = double_conv_3d(64 + 96, 64)
        self.dconv_up1 = double_conv_3d(64 + 32, 32)

        self.conv_last = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        conv1 = self.dconv_down1(x)

        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)

        x = self.maxpool(conv2)
        x = self.dconv_down3(x)


        x = self.upsample(x)

        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

class Conv3DFilter(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=[(3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)],
        padding_sizes=None,
        custom_bias=0,
    ):
        super(Conv3DFilter, self).__init__()
        self.custom_bias = custom_bias
        
        self.bias = 1e-8
        # Convolutional layer with padding to maintain input spatial dimensions
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size[0],
                        padding=padding_sizes[0],
                    ),

                    nn.ReLU(),
                    nn.Conv3d(
                        out_channels,
                        out_channels,
                        kernel_size[0],
                        padding=padding_sizes[0],
                    ),

                    nn.ReLU(),
                )
            ]
        )
        for kernel, padding in zip(kernel_size[1:-1], padding_sizes[1:-1]):

            self.convs.extend(
                [
                    nn.Sequential(
                        nn.Conv3d(
                            out_channels, out_channels, kernel, padding=padding
                        ),

                        nn.ReLU(),
                        nn.Conv3d(
                            out_channels, out_channels, kernel, padding=padding
                        ),

                        nn.ReLU(),
                    )
                ]
            )
        self.output_conv = nn.Conv3d(
            out_channels, 1, kernel_size[-1], padding=padding_sizes[-1]
        )
            
        # self.m = nn.LeakyReLU(0.1)

    def forward(self, input):
        x = input.unsqueeze(1)

        for module in self.convs:
            x = module(x) + x
        x = self.output_conv(x)
        x = torch.sigmoid(x).squeeze(1)
        return x
