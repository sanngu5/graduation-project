import os
import cv2
import imageio
import math
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

result_path = './test_results/'
compare_path = '../../../dataset/test/imgs/Y/'
n_frame = 125
pixel_max = 255
video_num = 50

def dataset():
	result = np.empty((video_num, 125, 128, 128, 3), dtype=np.float32)
	compare = np.empty((video_num, 125, 128, 128, 3), dtype=np.float32)
	
	for n in range(video_num):
		for i in range(125):
			result_file = os.path.join(result_path,'{}'.format(n),'output','{:03d}.png'.format(i+1))
			result_frame = np.array(Image.open(result_file).convert('RGB'))
			result[n, i] = result_frame
			
			compare_file = os.path.join(compare_path,'Y{}'.format(n),'{:03d}.png'.format(i+1))
			compare_frame = np.array(Image.open(compare_file).convert('RGB'))
			compare[n, i] = compare_frame
		
	return result, compare

def MSE(result, compare):
	mse = np.square(np.subtract(result, compare)).mean()
	
	return mse

def PSNR(mse):
	if mse == 0:
		return 100
	
	psnr = 20 * math.log10(pixel_max / math.sqrt(mse))
	
	return psnr

def SSIM(result, compare):
	result_ch = np.transpose(result, (0, 3, 1, 2))
	compare_ch = np.transpose(compare, (0, 3, 1, 2))
	ssim = 0
	for i in range(125):
		for j in range(3):
			ux = np.mean(result_ch[i,j])
			uy = np.mean(compare_ch[i,j])
			vx = np.var(result_ch[i,j])
			vy = np.var(compare_ch[i,j])
			cxy = 0
			for x in range(128):
				for y in range(128):
					tmp = (result_ch[i,j,x,y] - ux) * (compare_ch[i,j,x,y] - uy)
					cxy = cxy + tmp
			cxy = cxy / (128 * 128)
			c1 = (0.01 * pixel_max) ** 2
			c2 = (0.03 * pixel_max) ** 2
			score1 = (2 * ux * uy + c1)
			score2 = (2 * cxy + c2)
			score3 = (ux * ux + uy * uy + c1)
			score4 = (vx + vy + c2)
			score = (score1 * score2) / (score3 * score4)
			ssim = ssim + score
			
	ssim = ssim / 375
	
	return ssim
	
if __name__ == '__main__':
	result, compare = dataset()
	
	mse_score = 0
	psnr = 0
	ssim_score = 0
	
	for n in tqdm(range(video_num)):
		mse = MSE(result[n], compare[n])
		mse_score = mse_score + mse
		psnr = psnr + PSNR(mse)
		ssim_score = ssim_score + SSIM(result[n], compare[n])
		
	mse = mse / video_num
	psnr = psnr / video_num
	ssim_score = ssim_score / video_num
	
	print('MSE  : {:.4f}'.format(mse))
	print('PSNR : {:.4f}'.format(psnr))
	print('SSIM : {:.4f}'.format(ssim_score))