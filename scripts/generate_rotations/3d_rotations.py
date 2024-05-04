from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import imageio
import numpy as np
import SimpleITK as sitk
import sys
from PIL import Image
sys.path.append('.')
sys.path.append('../')
import transformations as T
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 
import imageio
import numpy
import plotly.graph_objects as go
from scipy import ndimage 
import nibabel as nib 


def gif_maker(gif_name,png_dir,gif_indx,num_gifs,dpi=90):
    # make png path if it doesn't exist already
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    # save each .png for GIF
    # lower dpi gives a smaller, grainier GIF; higher dpi gives larger, clearer GIF
    plt.savefig(png_dir+'frame_'+str(gif_indx)+'_.png',dpi=dpi)
    plt.close('all') # comment this out if you're only updating the x,y data

    if gif_indx==num_gifs-1:
        # sort the .png files based on index used above
        images,image_file_names = [],[]
        for file_name in os.listdir(png_dir):
            if file_name.endswith('.png'):
                image_file_names.append(file_name)       
        sorted_files = sorted(image_file_names, key=lambda y: int(y.split('_')[1]))

        # define some GIF parameters
        
        frame_length = 0.5 # seconds between frames
        end_pause = 4 # seconds to stay on last frame
        # loop through files, join them to image array, and write to GIF called 'wind_turbine_dist.gif'
        for ii in range(0,len(sorted_files)):       
            file_path = os.path.join(png_dir, sorted_files[ii])
            if ii==len(sorted_files)-1:
                for jj in range(0,int(end_pause/frame_length)):
                    images.append(imageio.imread(file_path))
            else:
                images.append(imageio.imread(file_path))
        # the duration is the time spent on each image (1/duration is frame rate)
        imageio.mimsave(gif_name, images,'GIF',duration=frame_length)


###############################################################################
# Dataset Locations

NIFTI_ROOT  = '/vol/medic01/users/bh1511/DATA_RAW/AliceBrainReconsAligned/test/'
SAVE_DIR    = ''

n_rotations = 100
n_offsets   = 16
max_z       = 40
###############################################################################

def rotation_matrix(axis, theta):
    
    #Generalized 3d rotation via Euler-Rodriguez formula, https://www.wikiwand.com/en/Euler%E2%80%93Rodrigues_formula
    #Return the rotation matrix associated with counterclockwise rotation about
    #the given axis by theta radians.
    
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def get_3d_locations(d,h,w,device_):
    locations_x = torch.linspace(0, w-1, w).view(1, 1, 1, w).expand(1, d, h, w)
    locations_y = torch.linspace(0, h-1, h).view(1, 1, h, 1).expand(1, d, h, w)
    locations_z = torch.linspace(0, d-1,d).view(1, d, 1, 1).expand(1, d, h, w)
    normalised_locs_x = (2.0*locations_x - (w-1))/(w-1)
    normalised_locs_y = (2.0*locations_y - (h-1))/(h-1)
    normalised_locs_z = (2.0*locations_z - (d-1))/(d-1)
    # stack locations
    locations_3d = torch.stack([normalised_locs_x, normalised_locs_y, normalised_locs_z], dim=4).view(-1, 3, 1).to(device_)
    return locations_3d

def rotate3d_tensor(input_tensor, rotation_matrix,mode='nearest',padding_mode="zeros"):
    """
    perform a 3d rotation to a pytorch  tensor of shape N C D H W according to a rotation matrix 
    """

    device_ = input_tensor.device
    bs,c, d, h, w  = input_tensor.shape
    
    # input_tensor = input_tensor.unsqueeze(0)
    # get x,y,z indices of target 3d data
    locations_3d = get_3d_locations(d, h, w, device_)
    locations_3d = locations_3d.double()
    # rotate target positions to the source coordinate

    rotated_3d_positions = torch.bmm(rotation_matrix.view(1, 3, 3).expand(d*h*w, 3, 3), locations_3d).view(1, d,h,w,3)

    rot_locs = torch.split(rotated_3d_positions, split_size_or_sections=1, dim=4)

    # change the range of x,y,z locations to [-1,1]

    grid = torch.stack([rot_locs[0], rot_locs[1], rot_locs[2]], dim=4).view(1, d, h, w, 3)
    # here we use the destination voxel-positions and sample the input 3d data trilinearly
    rotated_signal = nn.functional.grid_sample(input=input_tensor, grid=grid.repeat(bs,1,1,1,1), mode=mode,  align_corners=True,padding_mode=padding_mode)
    return rotated_signal#.squeeze(0)

###############################################################################
# Data Generation

SO3_GROUP   = SpecialOrthogonalGroup(3)

# Example 1 scan
database = '/work/ASNR-MICCAI-BraTS2023-GLI/'
root = '/work/output_3d_rotations/'

# loop through sub-folders to read flair scans (t2f)
for brain in glob.glob(database + '**/*t2f.nii.gz',recursive=True):

    brain_id = brain.split("/")[-1][:-7]
    print('Parsing:', brain_id)

    save_dir = root + brain_id 
    if not os.path.exists(save_dir):
        os.makedirs(root + brain_id)
    
    print("save dir:",save_dir)

    fixed_image_sitk_tmp    = sitk.ReadImage(brain, sitk.sitkFloat32)
    fixed_image_sitk        = sitk.GetImageFromArray(sitk.GetArrayFromImage(fixed_image_sitk_tmp))
    fixed_image_sitk        = sitk.RescaleIntensity(fixed_image_sitk, 0, 1)

    rotations   = np.pi * (np.random.rand(n_rotations, 3) - 0.5)
    IMG = sitk.GetArrayFromImage(fixed_image_sitk)
    
    centroid = ndimage.measurements.center_of_mass(IMG)
    
    # Create empty 3D tensor (240x240x240) and centre the ROI
    mask = np.zeros((IMG.shape[0],IMG.shape[0],IMG.shape[0]))
    
    z_offset = mask.shape[0]//2 - centroid[0]//IMG.shape[0]
    x_offset = mask.shape[1]//2 - centroid[1]//IMG.shape[1]
    y_offset = mask.shape[2]//2 - centroid[2]//IMG.shape[2]
    
    mask[int(z_offset- IMG.shape[0]//2): int(z_offset) + IMG.shape[0]//2, int(x_offset) - IMG.shape[1]//2: int(x_offset) + IMG.shape[1]//2, int(y_offset) - IMG.shape[2]//2: int(y_offset) + IMG.shape[2]//2] = IMG

    centred_IMG = torch.tensor(mask, dtype=torch.float64)
    centred_IMG = centred_IMG.unsqueeze(0).unsqueeze(0)

    axes = [
        [1,0,0],
        [0,0,1],
        [0,0,1]
    ]
    stack_tensors = []
    for axis in axes:
      
        axis = torch.tensor(axis).float()
        stack = [] 
        for theta in [0,360]:

            degree = theta
            theta = torch.tensor([theta * np.pi / 180])
            rot = rotation_matrix(axis, theta).double()
            Rotated_tensor = rotate3d_tensor(centred_IMG,rot)

            #  Conversion to Nifti File for Visualisation
            converted_array = np.array(Rotated_tensor, dtype=numpy.float32) 
            affine = numpy.eye(4)
            nifti_file = nib.Nifti1Image(converted_array[0,0,:,:,:], affine)
            print('saving')
            nib.save(nifti_file, save_dir + '/' + str(degree) + ".nii")