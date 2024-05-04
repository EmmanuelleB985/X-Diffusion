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
import glob 
import torch.nn as nn
from skimage.transform import resize
from math import exp
from skimage.transform import resize
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from convnet import ConvNet

class BratsDatasetModuleFromConfig(pl.LightningDataModule):
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
            
        image_transforms = []
        image_transforms = [torchvision.transforms.Resize((256,256))]
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

        self.image_transforms = torchvision.transforms.Compose(image_transforms)

        total_objects = len(next(os.walk(root_dir))[1])

        paths = []
        for s in glob.glob(root_dir + '**/*t2f.nii.gz',recursive=True):
            paths.append(s)
        
        self.val_paths = paths[math.floor(total_objects / 100. * 90.):] # used last 10% as validation
        self.train_paths = paths[:math.floor(total_objects / 100. * 80.)] # used first 80% as training

    def train_dataloader(self):
        dataset = BratsDataset(root_dir = self.train_paths, total_view=self.total_view, \
                                image_transforms=self.image_transforms)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        dataset = BratsDataset(root_dir = self.val_paths, total_view=self.total_view, \
                                image_transforms=self.image_transforms) 
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(BratsDataset(root_dir = self.test_paths, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class BratsDataset(Dataset):
    """
    Dataloader for reading nifti files of 3D brain rotations in range [0,360].
    Image resolution : [240, 240, 155]

    Args:
        -files (List[str]): list of paths to source images
        -transform (Callable): transform to apply to both source and target images
        -preload (bool): load all data when initializing the dataset
        
    Output: 
        -img shape:  (256, 256, 3)
        -target slice index (float) [-1,1]
        -axis of rotation i.e ["001", "010", "100"]
    """
    
    def __init__(self, root_dir='',
        image_transforms=[],
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=1
        ) -> None:


        self.files = root_dir
        self.tform = image_transforms

        if len(self.files) == 0:
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
        self.input = [] 
            
        for s in self.files:

            id = s.split('/')[-2]

            idx = random.sample(range(240), self.total_view + 1) 
            axes = ['100', '010', '001']
            axis = np.random.choice(axes, 2)
            
            self.rotation_axis.append(axis)
            
            img = nib.load(s).get_fdata(dtype=np.float32)
            
            #normalise img 
            sdata = (data-np.min(img))/(np.max(img)-np.min(img) + 1e-5)
                        
            dat = []
            
            # conditioning img
            for i in range(self.total_view):
                
                if axis[0] == "100":
                    data = sdata[idx[i],:,:] 
                elif axis[0] == "010":
                    data = sdata[:,idx[i],:]
                elif axis[0] == "001": 
                    id = random.sample(range(sdata.shape[2]), 1)[0]
                    data = sdata[:,:,id[0]]
                dat.append(data)
        
            # target img
            if axis[1] == "100":
                target = sdata[idx[-1],:,:] 
            elif axis[1] == "010": 
                target = sdata[:,idx[-1],:] 
            elif axis[1] == "001":
                target = sdata[:,:,idx[-1]]  
            
            self.depth.append(2*(idx[-1]/sdata.shape[0])-1) 
                
            dat = np.stack(dat)

            # multi-slice aggregation 
            avg_reduction = []
            for i in range(self.total_view):
                if i > 1: 
                    avgdot = torch.einsum("ij,kj->ik", torch.from_numpy(dat[i,:,:]).float(), torch.from_numpy(dat[i-1,:,:]).float())
                    avg_reduction.append(avgdot)
                    
            input_mdot = torch.stack(avg_reduction)    
            input_mdot = torch.mean(input_mdot,axis=0)
            img = input_mdot
    
            img = np.repeat(img[..., np.newaxis], 3, axis=2)
            img = Image.fromarray(np.uint8(img * 255.))
            img = self.tform(img)
        
            #normalise target 
            target = (target-np.min(target))/(np.max(target)-np.min(target) + 1e-5)
            target = np.repeat(target[..., np.newaxis], 3, axis=2)
            starget = Image.fromarray(np.uint8(target * 255.))
            starget = self.tform(starget)
            
            self.imgs.append(img)
            self.targets.append(starget)
            self.filenames.append(id)    
      
        print("total scans:",len(self.filenames))
        print("total number of files:",len(self.filenames))
    
         
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
     
        data = {}

        filename = self.filenames[idx]

        if self.return_paths:
            data["path"] = str(filename)

        data["image_target"] = self.targets[idx]
        data["image_cond"] = self.imgs[idx]
        data['filename'] = filename
        
        data["T"] = torch.tensor([self.rotation_axis[idx][1][0],self.rotation_axis[idx][1][1],self.rotation_axis[idx][1][2],self.depth[idx]]) 
        if self.postprocess is not None:
            data = self.postprocess(data)

        return data


if __name__ == '__main__':
    
    path_nifti = './ASNR-MICCAI-BraTS2023-GLI/'
    d2 = BratsDatasetModuleFromConfig(root_dir = path_nifti, batch_size = 1, total_view = 3, train=True)
    
    for batch in d2.train_dataloader():
        target = batch["image_target"]
        inp = batch["image_cond"]
        filename = batch["filename"]
        T_cond = batch['T']