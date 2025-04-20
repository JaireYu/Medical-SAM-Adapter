import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import json
from pycocotools import mask as Mask

from utils import random_box, random_click, random_gaussian_box


class DEMO(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training'):
        if mode == "Training":
            self.image_root = os.path.join(data_path, "test", "images")
            anno_file = os.path.join(data_path, "test", "annotations.json")
            self.annos = json.load(open(anno_file, 'r'))["annotations"]
        elif mode == "Test":
            self.image_root = os.path.join(data_path, "test", "images")
            anno_file = os.path.join(data_path, "test", "annotations.json")
            self.annos = json.load(open(anno_file, 'r'))["annotations"]
        self.mode = mode
        self.prompt = args.prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        anno = self.annos[index]
        image_id = anno["image_id"]
        img_path = os.path.join(self.image_root, "image_" + str(image_id) + ".png")
        img = Image.open(img_path).convert('RGB')
        
        rle_code = anno["segmentation"]
        rle_code['counts'] = str.encode(rle_code['counts'])
        mask = Mask.decode(rle_code)

        mask = Image.fromarray(mask * 255).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)
        elif self.prompt == 'box':
            box = random_gaussian_box(mask, deviation_rate = 0.1, max_offset=20)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        image_meta_dict = {'filename_or_obj':"image_" + str(image_id) + ".png"}
        if self.prompt == 'click':
            return {
                'image':img,
                'label': mask,
                'p_label': point_label,
                'pt': pt,
                'image_meta_dict':image_meta_dict,
            }
        elif self.prompt == 'box':
            return {
                'image':img,
                'label': mask,
                'box': box,
                'image_meta_dict':image_meta_dict
            }
        else:
            raise NotImplementedError