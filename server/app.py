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
from dataset import *
from network import MaskUNet, generator

import time

# CORS 작업
from flask_cors import CORS

# ## CONFIG


# ## HELPERS
###
def compress_nparr(nparr):
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed, len(uncompressed), len(compressed)

def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(zlib.decompress(bytestring)))



# ## MAIN SERVER DESCRIPTOR/ROUTINE

# Initialize the Flask application
app = Flask(__name__)

# Cross-Origin
CORS(
	app, 
    resources={r'*': {'origins': 'http://localhost:3000'}}, 
    supports_credentials=True)

# - POST로 받은 비디오 저장
# - (Frames로 변환)
# - (디캡션)
# - 성공여부 전송
@app.route('/upload', methods=['POST'])
def upload():
    global fn
    video = request.files['send']
    print('POST데이터: ', video.content_type)
    video.save('./video/{0}'.format(video.filename))
    vidcap = cv2.VideoCapture('./video/{0}'.format(video.filename))
    count = 1
    while True:
        success,image = vidcap.read()
        if not success:
            break
        print('Read a new frame: ',success)
        fname="{}.png".format("{:03d}".format(count))
        cv2.imwrite('./output/temp/'+fname,image)
        count+=1
    print("{} images are extracted in {}.".format(count,'./output/temp/'))
        
    #
    start = time.time()
    video_path='./output/temp'
    n_frames=125
    mask_model_path='./MaskExtractor.pth'
    model_G_path='./netG.pth'
    T=7
    s=3
    device = torch.device('cpu')

    masknet = MaskUNet(n_channels=3, n_classes=1)
    masknet.load_state_dict(torch.load(mask_model_path,map_location=device))
    masknet=masknet
    #masknet=torch.nn.DataParallel(masknet,device_ids=[0])
    masknet.eval()
    net_G=generator()
    net_G.load_state_dict(torch.load(model_G_path,map_location=device))
    net_G=net_G
    #net_G =torch.nn.DataParallel(net_G, device_ids=[0])    
    net_G.eval()
    frames = np.empty((125, 128, 128, 3), dtype=np.float32)
    for i in range(125):
        img_file = os.path.join(video_path,'{:03d}.png'.format(i+1))
        raw_frame = np.array(Image.open(img_file).convert('RGB'))/255
        frames[i] = raw_frame
    frames = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2)).copy()).float()
    frames=(frames-0.5)/0.5
    frames=frames.unsqueeze(0)
    with torch.no_grad():
        masks=masknet(frames)
        masks = (masks > 0.5).float()
        frames_padding=videopadding(frames,s,T) 
        masks_padding=videopadding(masks,s,T)  
        pred_imgs=[]
        for j in range(125):
            input_imgs=frames_padding[:,j:j+(T-1)*s+1:s]
            input_masks=masks_padding[:,j:j+(T-1)*s+1:s]
            pred_img= net_G(input_imgs,input_masks)
            pred=transforms.ToPILImage()(pred_img.squeeze(0)*0.5+0.5).convert('RGB')
            #pred.save('video_decaptioning/test_imgs1/%03d.png'%(j))
            pred_imgs.append(pred_img*0.5+0.5)
            
        vid=torch.cat(pred_imgs,dim=0)
        vid=(vid.cpu().numpy()*255).astype(np.uint8).transpose(0,2,3,1)
    pred_img = pred_imgs[0].numpy()
    imageio.mimwrite('./output/video.mp4',vid,fps=25,quality=8,macro_block_size=1)
#     def generate():
#         with open("./files/video.mp4", "rb") as f:
#             while True:
#                 chunk = ... # read each chunk or break if EOF

    print("time :", time.time() - start)
    # time.sleep(5000)

    return Response(response=video.filename, status=200, mimetype='text/plain')

# 디캡션 비디오 전송
@app.route('/download', methods=['GET'])
def download():
    # videoname = request.data
    # return send_file('./video/{0}'.format(video.filename), mimetype='video/mp4')
    return send_file('./output/video.mp4', mimetype='video/mp4')

###################################


# start flask app
app.run(host='0.0.0.0', port=5000)
