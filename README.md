# Pancreas Segmentation in CT Scan Images: Harnessing the Power of SAM
<p align="center">
  <img width="100%" src="Docs/ffpip.png">
</p>
In this repositpry we describe the code impelmentation of the paper: "Pancreas Segmentation in CT Scan Images: Harnessing the Power of SAM"

## Requirments
Frist step is install [requirements.txt](/requirements.txt) bakages in a conda eviroment.

## Dadaset and data loader description
For this segmentation report we used to populare pancreas datas:
- [NIH pancreas CT](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
- [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)

After downloading and allocating datasets, we used a specefice data format (.npy) and for this step [save.py](/data_handler/save.py) provided. `save_dir` and `labels_save_dir` should modify.

As defualt `data.py` and `data_loader_group.py` are used in the desired codes.  
