import argparse
import logging
import sys
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from model.network import UNet
from model.deeplabV3 import DeepLab
from tensorboardX import SummaryWriter
from dataset.dataset import *
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

dir_img = '../../../dataset/train/imgs'
dir_val='../../../dataset/dev/imgs'
m_root= './checkpoint'
model_path='./checkpoint/MaskExtractor3.pth'
save_path ='./test_mask/deeplab'
n_frames=5          ###5
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def eval_net(net,writer,epoch):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mse=0
    criterion = nn.MSELoss()
    val_data = VideoDecaptionData(dir_val,n_frames,False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    for step, (inputs,masks) in enumerate(val_loader):
        with torch.no_grad():
            imgs = inputs.cuda()
            true_masks = masks.cuda()
            pred_masks = net(imgs)
            pred_masks = torch.sigmoid(pred_masks)
            pred_masks = (pred_masks > 0.5).float() 

            #mask=pred_masks[:,0].squeeze(0).permute(1,2,0).cpu().numpy()*255     ##
            #cv2.imwrite(os.path.join('./masks', 'mask/%03d.png'%(0+1)), mask)      ## 
            mse+=criterion(pred_masks,true_masks)    
        if step==99:
            writer.add_scalar('Validation_loss', mse.item(), epoch+1)
            break
    return mse.item()

def test_net(net,writer,epoch):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mse=0
    criterion = nn.MSELoss()
    val_data = VideoDecaptionData(dir_val,n_frames,False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    for step, (inputs,masks) in enumerate(val_loader):
        with torch.no_grad():
            imgs = inputs.cuda()
            true_masks = masks.cuda()
            pred_masks = net(imgs)
            pred_masks = torch.sigmoid(pred_masks)
            pred_masks = (pred_masks > 0.5).float()
            for j in range(125):
                mask=pred_masks[j]
                p_mask=transforms.ToPILImage()(mask).convert('RGB')
                p_mask.save(os.path.join(save_path,'{:03d}.png'.format(j+1)))

            #mask=pred_masks[:,0].squeeze(0).permute(1,2,0).cpu().numpy()*255     ##
            #cv2.imwrite(os.path.join('./masks', 'mask/%03d.png'%(0+1)), mask)      ## 
            mse+=criterion(pred_masks,true_masks)    
        if step==1:
            #writer.add_scalar('Validation_loss', mse.item(), epoch+1)
            break

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,     ###128
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=bool, default=True,
                        help='Load model from a .pth file')

    return parser.parse_args() 

 
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    #net = UNet(n_channels=3, n_classes=1)
    net = DeepLab(1, in_channels=3, backbone='resnet50', pretrained=True, 
                output_stride=16, freeze_bn=False, freeze_backbone=False)
    if args.load:
        logging.info(f'Model loaded from {model_path}')
        net.load_state_dict(torch.load(model_path, map_location="cuda:0")) 
    net.cuda()
    net=torch.nn.DataParallel(net, device_ids=[0])
    try:
        train_data = VideoDecaptionData(dir_img, n_frames, True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        n_train = len(train_loader)
        logging.info(f'''Starting training:
                Epochs:          {args.epochs} 
                Batch size:      {args.batch_size}
                Learning rate:   {args.lr}
                Training size:   {n_train}
                Load:            {args.load}  
            ''')

        optimizer = optim.Adam(net.parameters(), lr=args.lr,betas=(0.9,0.999))
        criterion = nn.BCELoss()
        writer = SummaryWriter(comment=f'Mask_Extractor')
        valid_mse,valid_n=[],0
        for epoch in range(args.epochs):
            net.train()
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='video') as pbar:
                for global_step, (inputs,masks) in enumerate(train_loader):
                    #print(inputs.shape)
                    #print(masks.shape)
                    imgs = inputs.cuda()
                    #print(np.unique(masks))
                    true_masks = masks.cuda()
                    pred_masks = net(imgs)
                    #print(pred_masks.shape)#
                    loss = criterion(torch.sigmoid(pred_masks), true_masks)
                    #writer.add_scalar('Loss', loss.item(), global_step+epoch*n_train)
                    optimizer.zero_grad()
                    #torch.cuda.empty_cache()
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)
                    # if global_step%50 ==0:
                    #     mse=eval_net(net,writer,epoch)
                    #     #test_net(net,writer,epoch)
                    #     print(mse)
                    net.train()
                mse=eval_net(net,writer,epoch)
                print(mse)
                valid_mse.append(mse)
                if mse==min(valid_mse):
                    valid_n=0
                    torch.save(net.module.state_dict(),os.path.join(m_root,'MaskExtractor3.pth'))
                    logging.info(f'Model {epoch+1} saved !')
                else:
                    valid_n+=1
                if valid_n>9:
                    logging.info(f'Early Stopping!')
                    break       
        writer.close()
    except KeyboardInterrupt:
        torch.save(net.module.state_dict(),os.path.join(m_root,'MaskExtractor3.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
