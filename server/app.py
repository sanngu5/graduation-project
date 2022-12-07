import io
import zlib
import torch
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request, send_file, Response
import os
import cv2
import imageio
import numpy as np
import torch.nn.functional as F
import time
import tqdm
from dataset import *
from network import generator
from network import MaskUNet
from deeplabV3 import DeepLab
from flask_cors import CORS
from shutil import rmtree
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips

# Initialize the Flask application
app = Flask(__name__)

# Cross-Origin
CORS(
	app, 
    resources={r'*': {'origins': 'http://localhost:3000'}}, 
    supports_credentials=True)

# path
input_path = './input'

# mask_model_path='./checkpoints/MaskExtractor.pth'
# model_G_path='./checkpoints/netG.pth'

mask_model_path='./checkpoints/MaskExtractor_NewData.pth'
model_G_path='./checkpoints/netG_deeplab_NewData.pth'

frames_path='./output/temp'
chunks_path='./output/chunks'

# upload + reconstruction
@app.route('/upload', methods=['POST'])
def upload():
    # try:
    global fn
    video = request.files['send']
    
    print('POST데이터: ', video.content_type)
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    
    file_path = os.path.join(input_path, video.filename)
    video.save(file_path)
    vidcap = cv2.VideoCapture(file_path)

    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    count = 0
    while True:
        success,image = vidcap.read()
        if not success:
            break
        count += 1
        fname="{:03d}.png".format(count)
        cv2.imwrite(os.path.join(frames_path, fname), image)    
    print("{} images are extracted in {}.".format(count, frames_path))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda')
    start = time.time()

    masknet = DeepLab(n_channels=3, num_classes=1)
    # masknet = MaskUNet(n_channels=3, n_classes=1)
    masknet.load_state_dict(torch.load(mask_model_path,map_location=device))
    #masknet=masknet
    masknet=torch.nn.DataParallel(masknet,device_ids=[0])
    masknet.eval()

    net_G=generator()
    net_G.load_state_dict(torch.load(model_G_path,map_location=device))
    #net_G=net_G
    net_G =torch.nn.DataParallel(net_G, device_ids=[0])    
    net_G.eval()
    
    n_frames = 125
    T = 7
    s = 3

    n_total_frames = len(os.listdir(frames_path))
    n_chunks = (n_total_frames // 125)

    for step in range(n_chunks):
        # 마지막 청크 사이즈
        if step == n_chunks:
            n_frames = n_total_frames % 125
        if n_frames == 0:
            n_frames = 125

        frames = np.empty((n_frames, 128, 128, 3), dtype=np.float32)

        for i in range(n_frames):
            file_index = i + (125 * step) + 1
            img_file = os.path.join(frames_path,'{:03d}.png'.format(file_index))
            raw_frame = np.array(Image.open(img_file).convert('RGB'))/255
            frames[i] = raw_frame
        frames = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2)).copy()).float()
        frames=(frames-0.5)/0.5
        frames=frames.unsqueeze(0)

        with torch.no_grad():
            masks=masknet(frames)
            masks = (masks > 0.5).float()
            frames_padding=videopadding(frames,s,T).cuda()
            masks_padding=videopadding(masks,s,T).cuda()
            pred_imgs=[]

            for j in range(125):
                input_imgs=frames_padding[:,j:j+(T-1)*s+1:s]
                input_masks=masks_padding[:,j:j+(T-1)*s+1:s]
                pred_img= net_G(input_imgs,input_masks)
                pred_imgs.append(pred_img*0.5+0.5)
                
            vid=torch.cat(pred_imgs,dim=0)
            vid=(vid.cpu().numpy()*255).astype(np.uint8).transpose(0,2,3,1)

        if not os.path.exists(chunks_path):
            os.makedirs(chunks_path)
        chunk_path = os.path.join(chunks_path, "temp{0}.mp4".format(step))
        imageio.mimwrite(chunk_path, vid, fps=25, quality=8, macro_block_size=1)
        print("step{0}/{1} clear".format(step+1, n_chunks))
    
    chunks = []
    for index, chunk in enumerate(os.listdir(chunks_path)):
        chunk_path = os.path.join(chunks_path, f'temp{index}.mp4')
        chunks.append(VideoFileClip(chunk_path))
    final = concatenate_videoclips(chunks).set_fps(25)
    final.write_videofile('./output/video.mp4')

    print("time :", time.time() - start)
    rmtree(frames_path)
    # rmtree(chunks_path)
    return Response(response=video.filename, status=200, mimetype='text/plain')
    # except:
    #     print("변환 중 에러 발생!")
    #     if os.path.exists(frames_path): rmtree(frames_path)
    #     if os.path.exists(chunks_path): rmtree(chunks_path)
    #     return Response(response='변환 중 에러 발생!', status=500, mimetype='text/plain')


# 디캡션 비디오 전송
@app.route('/download', methods=['GET'])
def download():
    return send_file('./output/video.mp4', mimetype='video/mp4')

# start flask app
app.run(host='0.0.0.0', port=5000)
