# this is the implementation of the paper 'self-adaptively learning to demoire from focused and defocused image pairs (NeurIPS 2020)'
# Codes are based on the paper - Neural Blind Deconvolution Using Deep Priors (CVPR 2020)
# Liping Bao 2021.01.13 ###
#
# the parameters are not optimal. please feel free to tune them.

from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from networks.skip import skip
from networks.fcn import fcn
import cv2
import torch
import torch.optim
import glob
from skimage.io import imread
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM
import pdb
from PIL import Image
import torch.nn.functional as F
import skimage
import skimage.measure

# parameters setting
parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=2999, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[5, 5], help='size of blur kernel [height, width]')
parser.add_argument('--datablur_path', type=str, default="datasets/synscreenmoire/srblur", help='path to blurry image')
parser.add_argument('--datamoire_path', type=str, default="datasets/synscreenmoire/srmoire", help='path to moire image')
parser.add_argument('--datagt_path', type=str, default="datasets/synscreenmoire/srgt", help='path to gt image')
parser.add_argument('--save_path', type=str, default="results/synscreenmoire/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=100, help='lfrequency to save results')
opt = parser.parse_args()
#print(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore")

# list the blur gt and the moire images
files_blur = glob.glob(os.path.join(opt.datablur_path, '*.png'))
files_blur.sort()

files_gt = glob.glob(os.path.join(opt.datagt_path, '*.png'))
files_gt.sort()

files_moire = glob.glob(os.path.join(opt.datamoire_path, '*.png'))
files_moire.sort()

#pdb.set_trace()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)
recordfile = open('result.txt','w')
recordfile.close()


# start training !
for itemk,f in enumerate(files_blur):
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.0001  #0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    
    #pdb.set_trace()
    
    _, imgs = get_image(path_to_image, -1) 
    y = torch.Tensor(imgs).unsqueeze(0).detach()

    img_size = imgs.shape
    print(imgname)
    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw

    # load the images and convert to np.
    gt_path = files_gt[itemk]
    gt = Image.open(gt_path)
    gt = gt.resize((256,256),Image.BICUBIC)
    gt= np.array(gt)
    
    input_depth = 3
    moire_path = files_moire[itemk]
    moire = Image.open(moire_path)
    moire = moire.resize((256,256),Image.BICUBIC)
    moire= np.array(moire).astype(np.float32)
    moire = torch.tensor(moire).permute(2,0,1).unsqueeze(0)/255.
    
    blur_path = files_blur[itemk]
    blur = Image.open(blur_path)
    blur = blur.resize((256,256),Image.BICUBIC)
    blur= np.array(blur).astype(np.float32)
    blur = torch.tensor(blur).permute(2,0,1).unsqueeze(0)/255.
    
    
    # set the input of the G
    net_input = moire.detach()
    
    # depend on the size of the blur kernel:K//2-1
    #dim=(3,3,3,3)
    dim=(2,2,2,2)
    
    net_input =F.pad(net_input,dim,"constant",value=0)
    blur = F.pad(blur,dim,"constant",value=0)
    
    # ablation study for 'Z':
    #net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype).detach()
  
    #define the network G
    net = skip( input_depth, 3,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.cuda() 

    #define the network FCN
    n_k = 200
    #pdb.set_trace()
    net_input_kernel = get_noise(n_k, INPUT, (1, 1))
    net_input_kernel = net_input_kernel.squeeze().detach()

    net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
    net_kernel = net_kernel.cuda()

    # losses
    mse = torch.nn.MSELoss()
    
    ssim = SSIM()
    mse = mse.cuda()
    ssim = ssim.cuda()

    # optimizer
    optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()

    
    recordfile = open('result.txt','r+')
    recordfile.read()
    recordfile.write('%s\n'%imgname)
    recordfile.close()
    
    index = 0
    for step in tqdm(range(num_iter)):
        index = index +1
        
        # input regularization
        net_input = net_input_saved #
        
        # for ablation study 'Z', please use the following code:
        # net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()   

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        #out_x = net(net_input.cuda())
        out_parameter = net(net_input.cuda())       
        out_x = out_parameter
        
        # for the ablation study '\alpha*M+\beta', please use the following code:
        #out_x = out_parameter[:,0:3,:,:]*blur.cuda().detach()+out_parameter[:,3:6,:,:]
        
        #out_x = F.sigmoid(out_x)
        #net_input[:,3:6,:,:].cuda().detach()+out_parameter[:,6:9,:,:]
       
        
        
        out_k = net_kernel(net_input_kernel.cuda())
        #pdb.set_trace()
        out_k_m = out_k.view(-1,1,opt.kernel_size[0],opt.kernel_size[1])
        
        # for the ablation study 'alter', please use the following code:
        #if index%2==0:
            #out_x = out_x.detach()
        #else:
            #out_k_m = out_k_m.detach()
        
        # print(out_k_m)
        out_y1 = nn.functional.conv2d(out_x[:,0:1,:,:], out_k_m, padding=0, bias=None)
        out_y2 = nn.functional.conv2d(out_x[:,1:2,:,:], out_k_m, padding=0, bias=None)
        out_y3 = nn.functional.conv2d(out_x[:,2:,:,:], out_k_m, padding=0, bias=None)
        out_y = torch.cat([out_y1,out_y2,out_y3],1)
        if step < 1000:
            total_loss = mse(out_y,y.cuda()) #+0.1*(1-ssim(out_x[:,:,padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]], moire.detach().cuda()) )
        else:
            
            total_loss = 1-ssim(out_y, y.cuda()) #+1-ssim(out_x[:,:,padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]], moire.detach().cuda())
            #pdb.set_trace()
        total_loss.backward()
        optimizer.step()

        # save the images
        if (step+1) % opt.save_frequency == 0:
            #print('Iteration %05d' %(step+1))

            save_path = os.path.join(opt.save_path, '%s.png'%imgname)
            out_x_np = torch_to_np(out_x)
            out_x_np = out_x_np.squeeze()
            out_x_np = out_x_np[:,padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
            #pdb.set_trace()
            
            out_img = np.clip(np.transpose(out_x_np,(1,2,0))*255,0,255).astype(np.uint8)
            Image.fromarray(out_img).save(save_path)
            #imsave(save_path, out_x_np)
            psnr = skimage.measure.compare_psnr(out_img,gt)
            ssim_ = skimage.measure.compare_ssim(out_img,gt,multichannel=True)
            recordfile = open('result.txt','r+')
            recordfile.read()
            recordfile.write('%04d, psnr:%f, ssim:%f\n'%(step,psnr,ssim_))
            recordfile.close()
            save_path = os.path.join(opt.save_path, '%s_k.png'%imgname)
            out_k_np = torch_to_np(out_k_m)
            out_k_np = out_k_np.squeeze()
            out_k_np /= np.max(out_k_np)
            

            #torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))
            #torch.save(net_kernel, os.path.join(opt.save_path, "%s_knet.pth" % imgname))
