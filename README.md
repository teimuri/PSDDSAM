# Pancreas Segmentation in CT Scan Images: Harnessing the Power of SAM
<p align="center">
  <img width="100%" src="Docs/ffpip.jpg">
</p>
In this repositpry we describe the code impelmentation of the paper: "Pancreas Segmentation in CT Scan Images: Harnessing the Power of SAM"

## Requirments
Frist step is install [requirements.txt](/requirements.txt) bakages in a conda eviroment.

Clone the [SAM](https://github.com/facebookresearch/segment-anything) repository.

Use the code below to download the suggested checkpoint of SAM:
```
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
```

## Dadaset and data loader description
For this segmentation report we used to populare pancreas datas:
- [NIH pancreas CT](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
- [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)

After downloading and allocating datasets, we used a specefice data format (.npy) and for this step [save.py](/data_handler/save.py) provided. `save_dir` and `labels_save_dir` should modify.

As defualt `data.py` and `data_loader_group.py` are used in the desired codes.

Address can be modify in [args.py](args.py). 

Due to anonymous code submitiom we haven't share our Model Weights.  

## Train model
For train model we use the files [fine_tune_good.py](fine_tune_good.py) and [fine_tune_good_unet.py](fine_tune_good_unet.py) and the command bellow is an example for start training with some costume settings.
```
python3 fine_tune_good_unet.py --sample_size 66 --accumulative_batch_size 4 --num_epochs 60 --num_workers 8 --batch_step_one 20 --batch_step_two 30 --lr 3e-4 --inference

```
## Inference Model
To infrence both types of decoders just run the [double_decoder_infrence.py](double_decoder_infrence.py)

To get individually infrence SAM with or without prompt use [Inference_individually.py](Inference_individually)

## 3D Aggregator

To run the `3D Aggregator` codes are available in [kernel](/kernel) folder and just run the [run.sh](kernel/run.sh) file.

becuase of opening so many files, the `u -limit` thresh hold should be increased using:

```
u -limit 15000

```

