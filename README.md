# Deep_Video_Decaptioning
![teaser](https://github.com/Linya-lab/Video_Decaptioning/blob/master/images/teaser.png?raw=true)
## [Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0651.pdf)

## Citation
If any part of our paper and repository is helpful to your work, please generously cite with:
```
@inproceedings{chu2021deep,
  title={Deep Video Decaptioning},
  author={Chu Pengpeng, Quan Weize, Wang Tong, Wang Pan, Ren Peiran and Yan Dong-Ming},
  booktitle = {The Proceedings of the British Machine Vision Conference (BMVC)},
  year={2021}
}
```

## Introduction
![network](https://github.com/Linya-lab/Video_Decaptioning/blob/master/images/network.png?raw=true)

## Preparation
1. Install environment
```
conda env create -f environment.yml 
conda activate dvd
```
2. Install Dependencies
  - ffmpeg (video to png)

3. Install pretrained weight
  - [Mask_Extraction](https://maildhueducn-my.sharepoint.com/:u:/g/personal/2191420_mail_dhu_edu_cn/EaSYKsCiFoJBidxdfezACGsB4CfYak0hR_cGypUf9uN31A?e=aDCG3B)
  - [Video_Decaption](https://maildhueducn-my.sharepoint.com/:u:/g/personal/2191420_mail_dhu_edu_cn/EXQm-bYasU5Ag3221LoPAp8BBY7kOwyWfqlKAsCfBOnjZw?e=lsfMl6)

## Brief code instruction
1. Extract png files for each mp4 videos (use ./dataset/video2png.sh)
2. Set root path (modify --root_path flag in files)
* Note that we attached pretrained weight of the final model at onedrive.
  Please properly modify the path of pretrained weight for testing.
