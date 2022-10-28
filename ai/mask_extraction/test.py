import os
import cv2
import math
import imageio
import logging
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from dataset.dataset import *
from model.network import UNet
from torch.utils.data import DataLoader
from thop import profile
from thop import clever_format
from model.deeplabV3 import DeepLab

video_path='../../../dataset/test/imgs/X/X0'
save_path ='./test_mask/deeplab'
n_frames=125
mask_model_path='./checkpoint/MaskExtractor2.pth'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    masknet = DeepLab(1, in_channels=3, backbone='resnet50', pretrained=True, 
                output_stride=16, freeze_bn=False, freeze_backbone=False)
    masknet.load_state_dict(torch.load(mask_model_path))
    masknet=masknet.cuda()
    masknet=torch.nn.DataParallel(masknet,device_ids=[0])
    masknet.eval()
    frames = np.empty((125, 128, 128, 3), dtype=np.float32)
    for i in range(125):
        img_file = os.path.join(video_path,'{:03d}.png'.format(i+1))
        raw_frame = np.array(Image.open(img_file).convert('RGB'))/255
        frames[i] = raw_frame
    frames = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2)).copy()).float().cuda()
    frames=(frames-0.5)/0.5
    #frames=frames.unsqueeze(0)
    with torch.no_grad():
        masks=masknet(frames).squeeze(0)
        masks = (masks > 0.5).float()
        for j in range(125):
            mask=masks[j]
            p_mask=transforms.ToPILImage()(mask).convert('RGB')
            p_mask.save(os.path.join(save_path,'{:03d}.png'.format(j+1)))
