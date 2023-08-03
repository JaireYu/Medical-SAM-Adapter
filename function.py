
import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from conf import settings
import time
import cfg
from conf import settings
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
import torch
from einops import rearrange
import pytorch_ssim
import models.sam.utils.transforms as samtrans

# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)


import torch


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()

def set_drop_eval(m):
    classname = m.__class__.__name__
    if classname.find('Drop') != -1:
        m.eval()

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    if args.set_bn_eval:
        net.apply(set_bn_eval)
    if args.set_drop_eval:
        net.apply(set_drop_eval)
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            name = pack['image_meta_dict']['filename_or_obj']
            if args.prompt == 'click':
                if 'pt' not in pack:
                    imgs, pt, masks = generate_click_prompt(imgs, masks)
                else:
                    pt = pack['pt']
                    point_labels = pack['p_label']
                showp = pt
            elif args.prompt == 'box':
                assert 'box' in pack
                box = pack['box']
                showbox = box

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if args.prompt == 'point' and point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                # coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                coords_torch, labels_torch = coords_torch.unsqueeze(1), labels_torch.unsqueeze(1)
                pt = (coords_torch, labels_torch)
            elif args.prompt == 'box':
                box = torch.as_tensor(box, dtype=torch.float, device=GPUdevice)
                box = box.unsqueeze(1)
            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
            '''Train'''
            for n, value in net.image_encoder.named_parameters():
                if "Adapter" not in n:
                    value.requires_grad = False

            origin_imgs = imgs
            imgs = net.preprocess(imgs)
            imge= net.image_encoder(imgs)

            with torch.no_grad():
                # imge= net.image_encoder(imgs)
                if args.prompt == 'click':
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                elif args.prompt == 'box':
                    se, de = net.prompt_encoder(
                        points=None,
                        boxes=box,
                        masks=None,
                    )
                else:
                    raise NotImplementedError
            pred, _ = net.mask_decoder(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
              )

            loss = lossfunc(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    name_list = []
                    for na in name:
                        img_name = na.split('/')[-1].split('.')[0]
                        name_list.append(img_name)
                    if args.prompt == 'click':
                        vis_image(origin_imgs/255,pred,masks, os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' +str(epoch) + '+' + str(ind) + '.jpg'), reverse=False, points=showp)
                    elif args.prompt == 'box':
                        vis_image_box(origin_imgs/255,pred,masks, os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' +str(epoch) + '+' + str(ind) + '.jpg'), reverse=False, boxes=showbox)
                    else:
                        raise NotImplementedError
            pbar.update()

    return epoch_loss / len(train_loader)

def validation_sam(args, val_loader, epoch, threshold: Tuple, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader) # number of dataloader
    n_dataset_size = len(val_loader.dataset)  # the number of dataset
    iou_res = {}
    dice_res = {}
    for th in threshold:
        iou_res[str(th)] = 0
        dice_res[str(th)] = 0
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    #threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            cur_bsz = imgsw.shape[0]
            name = pack['image_meta_dict']['filename_or_obj']
            if args.prompt == 'click':
                if 'pt' not in pack:
                    imgs, pt, masks = generate_click_prompt(imgs, masks)
                else:
                    pt = pack['pt']
                    point_labels = pack['p_label']
                showp = pt
            elif args.prompt == 'box':
                assert 'box' in pack
                box = pack['box']
                showbox = box
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if args.prompt == 'click' and point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    # coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    coords_torch, labels_torch = coords_torch.unsqueeze(1), labels_torch.unsqueeze(1)
                    pt = (coords_torch, labels_torch)
                elif args.prompt == 'box':
                    box = torch.as_tensor(box, dtype=torch.float, device=GPUdevice)
                    box = box.unsqueeze(1)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                origin_imgs = imgs
                with torch.no_grad():
                    imgs = net.preprocess(imgs)
                    imge= net.image_encoder(imgs)

                    if args.prompt == 'click':
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                    elif args.prompt == 'box':
                        se, de = net.prompt_encoder(
                            points=None,
                            boxes=box,
                            masks=None,
                        )
                    else:
                        raise NotImplementedError

                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )

                    # tot += lossfunc(pred, masks) #previous version will overcalculate the weight of last incomplete batch
                    tot += lossfunc(pred, masks) * cur_bsz

                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        name_list = []
                        for na in name:
                            img_name = na.split('/')[-1].split('.')[0]
                            name_list.append(img_name)
                        if args.prompt == 'click':
                            vis_image(origin_imgs/255,pred,masks, os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' +str(epoch) + '+' + str(ind) + '.jpg'), reverse=False, points=showp)
                        elif args.prompt == 'box':
                            vis_image_box(origin_imgs/255,pred,masks, os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' +str(epoch) + '+' + str(ind) + '.jpg'), reverse=False, boxes=showbox)
                        else:
                            raise NotImplementedError
                    

                    iou_list, dice_list = eval_seg(pred, masks, threshold)
                    for iou, dice, th in zip(iou_list, dice_list, threshold):
                        iou_res[str(th)] += iou * cur_bsz
                        dice_res[str(th)] += dice * cur_bsz

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)
    for th in threshold:
        iou_res[str(th)] /= n_dataset_size
        dice_res[str(th)] /= n_dataset_size

    return tot/n_dataset_size, (iou_res, dice_res)

def inference_sam(args, val_loader, epoch, threshold: Tuple, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader) # number of dataloader
    n_dataset_size = len(val_loader.dataset)  # the number of dataset
    iou_res = {}
    dice_res = {}
    for th in threshold:
        iou_res[str(th)] = 0
        dice_res[str(th)] = 0
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    #threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            cur_bsz = imgsw.shape[0]
            name = pack['image_meta_dict']['filename_or_obj']
            if args.prompt == 'click':
                if 'pt' not in pack:
                    imgs, pt, masks = generate_click_prompt(imgs, masks)
                else:
                    pt = pack['pt']
                    point_labels = pack['p_label']
                showp = pt
            elif args.prompt == 'box':
                assert 'box' in pack
                box = pack['box']
                showbox = box
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if args.prompt == 'click' and point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    # coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    coords_torch, labels_torch = coords_torch.unsqueeze(1), labels_torch.unsqueeze(1)
                    pt = (coords_torch, labels_torch)
                elif args.prompt == 'box':
                    box = torch.as_tensor(box, dtype=torch.float, device=GPUdevice)
                    box = box.unsqueeze(1)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                origin_imgs = imgs
                with torch.no_grad():
                    imgs = net.preprocess(imgs)
                    imge= net.image_encoder(imgs)

                    if args.prompt == 'click':
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                    elif args.prompt == 'box':
                        se, de = net.prompt_encoder(
                            points=None,
                            boxes=box,
                            masks=None,
                        )
                    else:
                        raise NotImplementedError

                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )

                    pred_masks = net.postprocess_masks(pred, input_size=(h, w), original_size=(h, w))

                    # tot += lossfunc(pred, masks) #previous version will overcalculate the weight of last incomplete batch
                    tot += lossfunc(pred, masks) * cur_bsz

                    '''vis images'''
                    pred_masks = F.sigmoid(pred_masks)
                    best_thresh = args.inference_threshold
                    pred_masks[pred_masks > best_thresh] = 1
                    pred_masks[pred_masks <= best_thresh] = 0
                    for i in range(pred_masks.shape[0]):
                        pred_mask = pred_masks[i]
                        img_to_save = origin_imgs[i]
                        pred_mask = pred_mask.cpu().numpy()
                        img_to_save = img_to_save.cpu().numpy()
                        pred_mask = pred_mask.transpose(1,2,0).repeat(3, axis=2)
                        img_to_save = img_to_save.transpose(1,2,0)

                        img_to_save = pred_mask * 255 * 0.5 + img_to_save * 0.5
                        img_to_save = img_to_save.astype(np.uint8)
    
                        img_to_save = Image.fromarray(img_to_save)
                        img_to_save.save(os.path.join(args.path_helper['sample_path'], f'{ind}_{i}.png'))
                    

                    iou_list, dice_list = eval_seg(pred, masks, threshold)
                    for iou, dice, th in zip(iou_list, dice_list, threshold):
                        iou_res[str(th)] += iou * cur_bsz
                        dice_res[str(th)] += dice * cur_bsz

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)
    for th in threshold:
        iou_res[str(th)] /= n_dataset_size
        dice_res[str(th)] /= n_dataset_size

    return tot/n_dataset_size, (iou_res, dice_res)