""" train and test dataset

author jundewu
"""
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from utils import random_click
import random
import json
from monai.transforms import LoadImaged, Randomizable,LoadImage
from pycocotools import mask as Mask


class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):


        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        inout = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }

class Building_v1(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):


        if mode == 'Training':
            self.json_data = json.load(open(os.path.join(data_path, 'training_data.json')))
        elif mode == 'Test':
            self.json_data = json.load(open(os.path.join(data_path, 'testing_data.json')))
        self.mode = mode
        self.data_path = data_path
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        inout = 1
        point_label = 1

        """Get the images"""
        meta_json_info = self.json_data[index]
        name = meta_json_info["img"]
        img_path = os.path.join(self.data_path, 'img/' + name)
        
        mask_name = meta_json_info["mask"]
        msk_path = os.path.join(self.data_path, 'mask/' + mask_name)

        img = Image.open(img_path).convert('RGB')
        origin_x, origin_y = img.size

        rle_code = meta_json_info['rle_code']
        rle_code['counts'] = str.encode(rle_code['counts'])
        mask = Mask.decode(rle_code)

        mask = Image.fromarray(mask * 255).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.mode == "Test":
            pt = np.array(meta_json_info['pt'])
            pt[0] = pt[0] * self.img_size / origin_x
            pt[1] = pt[1] * self.img_size / origin_y
        elif self.prompt == 'click':
            try:
                pt = random_click(np.array(mask) / 255, point_label, inout)
                # when the mask is too small, there is no 255, because smooth when convert('L') and resize
            except ValueError:
                binary_mask = np.array(mask)
                binary_mask[binary_mask != 0] = 255
                pt = random_click(np.array(binary_mask) / 255, point_label, inout)
            except:
                pass
        else:
            raise NotImplementedError

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        #name = name.split('/')[-1].split(".jpg")[0]
        
        """
        # ATTENTION: fixed bug in it
        # pt: reverse (y,x) to (x,y)
        !!!!!!!!!!!!!!!!!!!
        """
        sam_pt = np.array([-1, -1])
        sam_pt[0] = pt[1]
        sam_pt[1] = pt[0]
        # bug fixed
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':sam_pt,
            'image_meta_dict':image_meta_dict,
        }
    
class Building_refined_v1(Dataset):
    def __init__(self, args, data_path, mode = 'Training',prompt = 'click', plane = False):
        if mode == 'Training':
            self.json_data = json.load(open(os.path.join(data_path, 'training_data.json')))
        elif mode == 'Test':
            self.json_data = json.load(open(os.path.join(data_path, 'testing_data.json')))
        self.mode = mode
        self.data_path = data_path
        self.prompt = prompt
        self.img_size = args.image_size

        if mode == 'Training':
            self.transform_flip = transforms.Compose([
                transforms.Resize((args.image_size,args.image_size)),
                transforms.RandomHorizontalFlip(args.horizontal_flip_prob),
                transforms.RandomVerticalFlip(args.vertical_flip_prob),
                transforms.ToTensor()
            ])
            self.transform_flip_mask = transforms.Compose([
                transforms.Resize((args.image_size,args.image_size)),
                transforms.RandomHorizontalFlip(args.horizontal_flip_prob),
                transforms.RandomVerticalFlip(args.vertical_flip_prob),
                transforms.ToTensor()
            ])
            self.transform_rotate = transforms.RandomApply([
                transforms.RandomRotation(degrees=args.rotate_degree)], p=args.rotate_prob
            )
            self.transform_resize_mask = transforms.Compose([
                transforms.Resize((args.out_size,args.out_size))
            ])
        else:
            self.transform_test = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
            ])
            self.transform_test_mask = transforms.Compose([
                transforms.Resize((args.out_size, args.out_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        inout = 1
        point_label = 1

        """Get the images"""
        meta_json_info = self.json_data[index]
        name = meta_json_info["img"]
        img_path = os.path.join(self.data_path, 'img/' + name)
        
        mask_name = meta_json_info["mask"]
        msk_path = os.path.join(self.data_path, 'mask/' + mask_name)

        img = Image.open(img_path).convert('RGB')
        origin_x, origin_y = img.size

        rle_code = meta_json_info['rle_code']
        rle_code['counts'] = str.encode(rle_code['counts'])
        mask = Mask.decode(rle_code)

        mask = Image.fromarray(mask * 255).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        if self.mode == "Test":
            # transform image
            state = torch.get_rng_state()
            img = self.transform_test(img)
            torch.set_rng_state(state)
            # transform mask
            mask = self.transform_test_mask(mask)
            pt = np.array(meta_json_info['pt'])
            pt[0] = pt[0] * self.img_size / origin_x
            pt[1] = pt[1] * self.img_size / origin_y
        elif self.prompt == 'click':
            # flip image
            state = torch.get_rng_state()
            img = self.transform_flip(img)
            torch.set_rng_state(state)
            # flip mask
            mask = self.transform_flip_mask(mask)
            # rotate img
            state = torch.get_rng_state()
            rotated_img = self.transform_rotate(img)
            torch.set_rng_state(state)
            # rotate mask
            rotated_mask = self.transform_rotate(mask)
            if rotated_mask.sum() != 0:
                # rotated operation may cause the mask to be all zero
                mask = rotated_mask
                # transform image
                img = rotated_img

            try:
                pt = random_click(np.array(mask[0]), point_label, inout)
                # when the mask is too small, there is no 255, because smooth when convert('L') and resize
            except ValueError:
                binary_mask = np.array(mask[0])
                binary_mask[binary_mask != 0] = 1
                pt = random_click(np.array(binary_mask), point_label, inout)
            except:
                pass
            mask = self.transform_resize_mask(mask)
        else:
            raise NotImplementedError

        #name = name.split('/')[-1].split(".jpg")[0]
        
        """
        # ATTENTION: fixed bug in it
        # pt: reverse (y,x) to (x,y)
        !!!!!!!!!!!!!!!!!!!
        """
        sam_pt = np.array([-1, -1])
        sam_pt[0] = pt[1]
        sam_pt[1] = pt[0]
        # bug fixed
        image_meta_dict = {'filename_or_obj':name}
        img = img * 255
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':sam_pt,
            'image_meta_dict':image_meta_dict,
        }