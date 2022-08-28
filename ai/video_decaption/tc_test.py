from nis import cat
import os
import cv2
import imageio
import numpy as np
import torch.nn.functional as F

from PIL import Image
from dataset.dataset import *
from model.network import MaskUNet,generator
from tqdm import tqdm

video_path='../../../dataset/test/imgs/X'
n_frames=125
mask_model_path='../mask_extraction/checkpoint/MaskExtractor.pth'
model_G_path='../video_decaption/checkpoint/netG_origin.pth'
T=7
s=3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

import time
if __name__ == '__main__':
    masknet = MaskUNet(n_channels=3, n_classes=1)
    masknet.load_state_dict(torch.load(mask_model_path))
    masknet=masknet.cuda()
    masknet=torch.nn.DataParallel(masknet,device_ids=[0])
    masknet.eval()
    net_G=generator()

    net_G.load_state_dict(torch.load(model_G_path))
    net_G=net_G.cuda()
    net_G =torch.nn.DataParallel(net_G, device_ids=[0])    
    net_G.eval()

    for h in tqdm(range(50)):
    # for h in tqdm(range(os.listdir(video_path).__len__())):
        frames = np.empty((125, 128, 128, 3), dtype=np.float32)
        for i in range(125):
            img_file = os.path.join(video_path, 'X{:d}'.format(h), '{:03d}.png'.format(i+1))
            raw_frame = np.array(Image.open(img_file).convert('RGB'))/255
            frames[i] = raw_frame
        frames = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2)).copy()).float().cuda()
        frames=(frames-0.5)/0.5
        frames=frames.unsqueeze(0)
        with torch.no_grad():
            masks=masknet(frames)
            masks = (masks > 0.5).float().cuda()
            frames_padding=videopadding(frames,s,T).cuda()  
            masks_padding=videopadding(masks,s,T).cuda()   
            pred_imgs=[]
            
            forward_pred_imgs = []
            backward_pred_imgs = []

            for j in range(125):
                input_imgs=frames_padding[:,j:j+(T-1)*s+1:s]
                input_masks=masks_padding[:,j:j+(T-1)*s+1:s]
                pred_img= net_G(input_imgs,input_masks)
                
                # 타겟프레임을 예측결과로 대치
                frames_padding[:,(j+((T-1)*s)//2)] = pred_img
                # 예측결과 저장
                forward_pred_imgs.append(pred_img)
            
            for j in range(125, 0, -1):
                input_imgs=frames_padding[:,j:j+(T-1)*s+1:s]
                input_masks=masks_padding[:,j:j+(T-1)*s+1:s]
                pred_img= net_G(input_imgs,input_masks)
                
                # 타겟프레임을 예측결과로 대치
                frames_padding[:, j+((T-1)*s)//2] = pred_img
                # 예측결과 저장
                backward_pred_imgs.append(pred_img)

            for j in range(125):
                pred_img = torch.add(forward_pred_imgs[j], backward_pred_imgs[j])
                pred_img = torch.div(pred_img, 2)
                pred_img = forward_pred_imgs[j]

                # 경로 생성, 이미지 스케일링, 저장
                test_imgs_path = './test_results/{:d}'.format(h)
                createDirectory(os.path.join(test_imgs_path, 'output'))
                createDirectory(os.path.join(test_imgs_path, 'mask'))
                pred=transforms.ToPILImage()(pred_img.squeeze(0)*0.5+0.5).convert('RGB')
                pred.save(os.path.join(test_imgs_path, 'output/%03d.png'%(j+1)))
                mask=masks[:,j].squeeze(0).permute(1,2,0).cpu().numpy()*255
                # 용량문제로 mask 저장은 보류
                # cv2.imwrite(os.path.join(test_imgs_path, 'mask/%03d.png'%(j+1)), mask)
                pred_imgs.append(pred_img*0.5+0.5)

            video=torch.cat(pred_imgs,dim=0)
            video=(video.cpu().numpy()*255).astype(np.uint8).transpose(0,2,3,1)
            imageio.mimwrite(os.path.join(test_imgs_path,'video.mp4'),video,fps=25,quality=8,macro_block_size=1)