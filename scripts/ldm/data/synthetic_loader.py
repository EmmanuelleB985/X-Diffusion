import math
import torch 
from random import random
import pandas as pd 
import sys
sys.path.append('.')
from typing import Dict
import webdataset as wds
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config 
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
from typing import Callable, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
from typing import List, Optional
from packaging import version
if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore
    
    
class SyntheticDatasetModuleFromConfig(pl.LightningDataModule):
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
        
        
        file = pd.read_csv(self.root_dir + "./synthetic_m.csv")
        file_list = file.loc[:,"Input"].to_list()
    
        total_objects = len([name for name in file_list])
            
        paths = [name for name in file.loc[:,"Input"]]

        self.val_paths = paths[math.floor(total_objects / 100. * 90):] 
        self.train_paths = paths[:math.floor(total_objects / 100. * 80)] 
        

    def train_dataloader(self):
        dataset = Synthetic_Dataset(root_dir = self.train_paths, total_view=self.total_view, \
                                image_transforms=self.image_transforms)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        dataset = Synthetic_Dataset(root_dir = self.val_paths, total_view=self.total_view, \
                                image_transforms=self.image_transforms) 
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class Synthetic_Dataset(Dataset):

    """
    Dataloader for reading .npy of synthetic volume rotation.
    Args:
        -files (List[str]): list of paths to source images
        -transform (Callable): transform to apply to both source and target images
        -preload (bool): load all data when initializing the dataset
        
    Output: 
        -img shape:  (256, 256, 3)
        -target slice index (float) [-1,1]
        -axis of rotation i.e ["001", "010", "100"]
    """
    
    def __init__(self, root_dir='/work/emmanuelle/',
        image_transforms=[],
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=1
        ) -> None:


        self.files = root_dir
        self.tform = image_transforms
        
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
        file = pd.read_csv(self.files + "synthetic.csv")


        for i in range(self.files):
            
            # load cone slices 
            input_filename = file.iloc[i]["Input"]
            target_filename = file.iloc[i]["Target"]
            
            axis = file.iloc[i]["Axis_Target"] 
            index_input = file.iloc[i]["Index_Input"]  
            index_target = file.iloc[i]["Index_Target"]  
            
            
            input_slice = np.load(input_filename)
            target_slice = np.load(target_filename)
    

            sdata = self.tform(Image.fromarray((input_slice*255).astype("uint8")))
            self.imgs.append(sdata)
            
            
            self.rotation_axis.append(literal_eval(axis))
        
            
            sdata_target = self.tform(Image.fromarray(((target_slice*255)).astype("uint8")))
            self.targets.append(sdata_target)
            self.depth.append(index_target) 
            
            input_filename = input_filename.split('/')[-1]
            
            self.filenames.append(input_filename)
            
                
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
        
        data["T"] = torch.tensor([self.rotation_axis[idx][0],self.rotation_axis[idx][1],self.rotation_axis[idx][2],self.depth[idx]]) # try with axis of rotation values 

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data


if __name__ == '__main__':
    
    path_nifti = '/work/ASNR-MICCAI-BraTS2023-GLI/'
    d2 = SyntheticDatasetModuleFromConfig(root_dir = 1, batch_size = 1, total_view = 1, train=True)
       
    for batch in d2.train_dataloader():
        target = batch["image_target"]
        inp = batch["image_cond"]
        filename = batch["filename"]
        T_cond = batch['T']
    


