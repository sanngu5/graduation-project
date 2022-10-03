import os
import torch
import numpy as np
import gc
import pyiqa

from PIL import Image
from tqdm import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pred_folder = './test_results/'
gt_folder = '../../../../dataset/test/imgs/Y/'
n_frame = 125

total_n_video = 500
batch_size = 1

# ms_ssim 제외 13개
metric_name_list= ['ssim', 'psnr', 'nlpd', 'pieapp', 'vif', 'vsi', 'ckdn', 'cw_ssim', 'dists', 'fsim', 'gmsd', 'lpips', 'mad']

if __name__ == '__main__':

    metric_list = []
    for metric_name in metric_name_list:
        metric_list.append(pyiqa.create_metric(metric_name, device=torch.device('cuda')))

    total_score_list = []   
    for step, start in enumerate(tqdm(range(0, total_n_video, batch_size))):
        # print("step:{:3}".format(step))
        end = start + batch_size

        batch_pred, batch_gt = [], []
        for i in range(start, end):
            for j in range(n_frame):
                file_pred = os.path.join(pred_folder,'{}'.format(i), 'output', '{:03d}.png'.format(j+1))
                file_gt = os.path.join(gt_folder, 'Y{}'.format(i), '{:03d}.png'.format(j+1))
                
                frame_pred = np.array(Image.open(file_pred).convert('RGB')).transpose(2, 0, 1)
                frame_gt = np.array(Image.open(file_gt).convert('RGB')).transpose(2, 0, 1)

                batch_pred.append(torch.Tensor(frame_pred) / 255.0)
                batch_gt.append(torch.Tensor(frame_gt) / 255.0)

            # gc.collect()
            # del frame_pred, frame_gt
    
        batch_pred = torch.stack(batch_pred)
        batch_gt = torch.stack(batch_gt)

        score_list = []
        for i, metric in enumerate(metric_list):
            score = metric(batch_pred, batch_gt).sum() / (n_frame*batch_size)
            score_list.append(score)
            # print('{0}: {1:.4f}'.format(metric_name_list[i], score))
        score_list = torch.stack(score_list)
        total_score_list.append(score_list)

        gc.collect()
        del batch_pred, batch_gt
    
    total_score_list = torch.stack(total_score_list)
    n_step = total_n_video / batch_size
    total_score_list = total_score_list.sum(dim=0) / n_step

    for i, total_score in enumerate(total_score_list):
        print('{0}: {1:.4f}'.format(metric_name_list[i], total_score))

        