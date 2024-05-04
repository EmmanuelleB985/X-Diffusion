import sys
sys.path.append('.')
from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
import nibabel as nib
from typing import Callable, List, Optional
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch
            
    
class UKBBDatasetModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation
            
        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []

        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
    
        total_objects = len([name for name in os.listdir(self.root_dir )])
            
        paths = [os.path.join(self.root_dir, name) for name in os.listdir(self.root_dir )]

        self.val_paths = paths[math.floor(total_objects / 100. * 90):] 

        self.train_paths = paths[:math.floor(total_objects / 100. * 80)] 
        
    def train_dataloader(self):
        dataset = UKBBDataset(root_dir = self.train_paths, total_view=self.total_view, \
                                image_transforms=self.image_transforms)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        dataset = UKBBDataset(root_dir = self.val_paths, total_view=self.total_view, \
                                image_transforms=self.image_transforms) 
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(UKBBDataset(root_dir = self.val_paths, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class UKBBDataset(Dataset):
    
    """
    Dataloader for reading .pkl files of whole-body MRIs. 
    Image resolution : [501, 224, 156]

    Args:
        -files (List[str]): list of paths to source images
        -transform (Callable): transform to apply to both source and target images
        -preload (bool): load all data when initializing the dataset
        
    Output: 
        -img shape:  (256, 256, 3)
        -target slice index (float) [-1,1]
        -axis of rotation i.e ["001", "010", "100"]
    """
    
    def __init__(self, root_dir='/work/emmanuelle/UKBiobank/Dixon-Mri-v2-stitched/',
        image_transforms=[],
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=1
        ) -> None:

        self.root_dir = root_dir
        
        self.tform = image_transforms

        if len(self.root_dir) == 0:
            raise ValueError(f'Number of source images must be non-zero')

        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess

        self.default_trans = default_trans
        self.return_paths = return_paths
        self.total_view = total_view

        self.imgs = []
        self.targets = []
        self.rotation_axis = []
        self.depth = [] 
        self.slice = []
        self.filenames = []
        
        volumes = [] 
        for mri_filename in self.root_dir:
            
            with open(mri_filename.replace('.zip','.pkl'), 'rb') as f:
                mri_dict = pickle.load(f)
            for idx, seq in enumerate(mri_dict['volumes']):
                vol = mri_dict['volumes']['F']
                volumes.append(vol)
                
                #pad to square array 
                sdata = np.zeros((512,512))
                idx_i = random.sample(range(224),1) 
            

                sdata[6:507,144:368] = vol[:,idx_i,:] # sagittal slice
                self.rotation_axis.append("010") # sagittal
            
                # Apply transforms 
                sdata = self.tform(sdata)
                self.imgs.append(sdata)
                
                starget = np.zeros((512,512))
                # target slice 
                idx_t = random.sample(range(156),1) 

                starget[6:507,144:368] = vol[:,idx_t[0],:] # sagittal mid slice

                # Apply transforms 
                sdata = self.tform(starget)
                
                self.targets.append(starget)
            
                self.depth.append(2*(idx_t[0]/224)-1)

                self.filenames.append(mri_filename)
  
            
                
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
     
        data = {}

        filename = self.filenames[idx]

        if self.return_paths:
            data["path"] = str(filename)

        data["image_target"] = self.targets[idx]
        data["image_cond"] = self.imgs[idx]

   
        data["T"] = torch.tensor([self.rotation_axis[idx][0][0],self.rotation_axis[idx][0][1],self.rotation_axis[idx][0][2],self.depth[idx]]) 

        if self.postprocess is not None:
           data = self.postprocess(data)

        return data


if __name__ == '__main__':
    
    path= './UKBiobank/Dixon-Mri-v2-stitched/'
        
    total_objects = len([name for name in os.listdir(path)])
    paths = [os.path.join(path, name) for name in os.listdir(path)]
    test = paths[math.floor(total_objects / 100. * 99):] # used last 1% as validation
        
    d2 = UKBBDataset(root_dir=test)
    
    for i in d2:
        print(i["image_target"])
        print(np.shape(np.array(i["image_cond"])))
        print((i["T"]))