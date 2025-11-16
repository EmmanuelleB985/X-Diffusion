from contextlib import nullcontext
import numpy as np
import gc
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from ldm.util import instantiate_from_config
import os 
import nibabel as nib
from ldm.data.bratsloader import BratsDatasetModuleFromConfig
from tqdm import tqdm
import argparse
import ldm.modules.diffusionmodules.openaimodel as openai_module

"""
Script based on Zero-123 https://github.com/cvlab-columbia/zero123
"""    
description = \
"""
Generate Novel View Synthetis given an Input Image using a Fine-Tuned version of Stable Diffusion trained on Objeverse Dataset by Zero-123 (https://github.com/cvlab-columbia/zero123)
. Stable diffusion weights can be obtained from [Lambda](https://lambdalabs.com/),trained by [Justin Pinkney](https://www.justinpinkney.com) ([@Buntworthy](https://twitter.com/Buntworthy)).
__Get the [code](https://github.com/justinpinkney/stable-diffusion) and [model](https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned).__
![](https://raw.githubusercontent.com/justinpinkney/stable-diffusion/main/assets/im-vars-thin.jpg)
"""

def load_model_from_config(config, ckpt, device, verbose=False):
    """Load the diffusion model from checkpoint - memory efficient version"""
    print(f"Loading model from {ckpt}")
    
    # CPU loading first
    print("Loading checkpoint to CPU...")
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd["state_dict"]
    
    # Delete the full checkpoint to free memory
    del pl_sd
    gc.collect()
    
    print("Instantiating model...")
    model = instantiate_from_config(config.model)
    
    print("Loading state dict...")
    m, u = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"Missing keys: {len(m)}, Unexpected keys: {len(u)}")
    
    # Clear state dict
    del sd
    gc.collect()
    
    # Move to GPU in eval mode with gradient checkpointing disabled
    print("Moving model to GPU...")
    model.eval()
    
    for module in model.children():
        module.to(device)
        torch.cuda.empty_cache()
    
    print("Model loaded successfully")
    return model


@torch.no_grad()
def sample_slice_with_denoising(input_im, T_cond, model, sampler, h, w, 
                                ddim_steps, n_samples, scale, ddim_eta, device):
    """
    Run the full denoising process for a single slice
    Returns the generated image after denoising
    """
    
    # Force model to device and ensure it stays there
    model = model.to(device)
    model.eval()
    
    # Override the sampler's model to ensure device consistency
    sampler.model = model
    
    original_timestep_embedding = openai_module.timestep_embedding
    
    def timestep_embedding_with_device(timesteps, dim, max_period=10000, repeat_only=False):
        result = original_timestep_embedding(timesteps, dim, max_period, repeat_only)
        return result.to(device)
    
    openai_module.timestep_embedding = timestep_embedding_with_device
    precision_scope = autocast if torch.cuda.is_available() else nullcontext
    
    with precision_scope("cuda"):
        with model.ema_scope():
            x = input_im
            if len(x.shape) == 3:
                x = x[..., None]
            x = rearrange(x, 'b h w c -> b c h w')
            x = x.to(memory_format=torch.contiguous_format).float()
            
            x = x.to(device)
            
            # Get CLIP embedding from input image
            c = model.get_learned_conditioning(x).tile(n_samples, 1, 1)
            
            # Prepare time/condition embedding 
            T = T_cond.to(device).to(memory_format=torch.contiguous_format).float()
            T = T[0][None, None, :].repeat(n_samples, 1, 1)
            
            # Combine embeddings 
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            
            cond = {}
            cond['c_crossattn'] = [c.to(device)]
            
            encoded = model.encode_first_stage(x).mode().detach()
            encoded = encoded.repeat(n_samples, 1, 1, 1)
            cond['c_concat'] = [encoded.to(device)]
            
            # Prepare unconditional conditioning for classifier-free guidance
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8, device=device)]
                uc['c_crossattn'] = [torch.zeros_like(c, device=device)]
            else:
                uc = None
            
            # Sample from diffusion model
            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=None
            )
            
            # Decode from latent space
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            output = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            
            return output


@torch.no_grad()
def generate_reconstruction(input_im, model, device):
    """
    Generate reconstruction (encode-decode without denoising)
    Useful for comparison
    """
    x = input_im
    if len(x.shape) == 3:
        x = x[..., None]
    x = rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float().to(device)
    
    with model.ema_scope():
        # Encode
        encoder_posterior = model.encode_first_stage(x)
        z = model.get_first_stage_encoding(encoder_posterior).detach()
        
        # Decode
        xrec = model.decode_first_stage(z)
        output = torch.clamp((xrec + 1.0) / 2.0, min=0.0, max=1.0)
        
        return output


def save_slice_as_png(tensor_slice, save_path):
    """Save a single slice as PNG image"""
    # tensor_slice shape: [C, H, W] or [H, W, C]
    if tensor_slice.shape[0] in [1, 3]:  # channels first
        img_array = 255. * rearrange(tensor_slice.cpu().numpy(), 'c h w -> h w c')
    else:  # channels last or single channel
        img_array = 255. * tensor_slice.cpu().numpy()
    
    if img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
    
    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(save_path)


def process_batch_to_3d_volume(dataloader, model, sampler, output_dir, 
                               h=256, w=256, ddim_steps=50, scale=7.5, 
                               ddim_eta=0.0, device='cuda:0', 
                               save_individual_slices=True,
                               generate_denoised=True,
                               generate_reconstruction=False):
    """
    Process all slices from dataloader and generate 3D volumes
    
    Args:
        generate_denoised: If True, run full denoising process
        generate_reconstruction: If True, generate encode-decode reconstruction
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for 3D volumes
    denoised_slices = []
    reconstructed_slices = []
    input_slices = []
    target_slices = []
    
    patient_name = None
    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        target = batch["image_target"]
        input_im = batch["image_cond"]
        filename = batch["filename"]
        T_cond = batch['T']
        
        # Get patient name from first batch
        if patient_name is None:
            patient_name = filename[0].split('_')[0] if isinstance(filename, list) else filename.split('_')[0]
        
        # Move to device
        input_im = input_im.to(device)
        target = target.to(device)
        T_cond = T_cond.to(device)
        
        # Generate denoised output
        if generate_denoised:
            denoised_output = sample_slice_with_denoising(
                input_im, T_cond, model, sampler, 
                h, w, ddim_steps, n_samples=1, 
                scale=scale, ddim_eta=ddim_eta, device=device
            )
            denoised_slices.append(denoised_output[0].cpu().numpy())
            
            # Save individual slice
            if save_individual_slices:
                slice_dir = os.path.join(output_dir, 'slices_denoised')
                os.makedirs(slice_dir, exist_ok=True)
                save_path = os.path.join(slice_dir, f"{filename[0]}_denoised.png")
                save_slice_as_png(denoised_output[0], save_path)
        
        # Generate reconstruction
        if generate_reconstruction:
            recon_output = generate_reconstruction(input_im, model, device)
            reconstructed_slices.append(recon_output[0].cpu().numpy())
            
            if save_individual_slices:
                slice_dir = os.path.join(output_dir, 'slices_reconstruction')
                os.makedirs(slice_dir, exist_ok=True)
                save_path = os.path.join(slice_dir, f"{filename[0]}_recon.png")
                save_slice_as_png(recon_output[0], save_path)
        
        # Store input and target for reference
        input_normalized = torch.clamp((input_im + 1.0) / 2.0, min=0.0, max=1.0)
        target_normalized = torch.clamp((target + 1.0) / 2.0, min=0.0, max=1.0)
        
        input_slices.append(input_normalized[0].cpu().numpy())
        target_slices.append(target_normalized[0].cpu().numpy())
        
        # Save input and target slices
        if save_individual_slices:
            input_dir = os.path.join(output_dir, 'slices_input')
            target_dir = os.path.join(output_dir, 'slices_target')
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(target_dir, exist_ok=True)
            
            save_slice_as_png(input_normalized[0], 
                            os.path.join(input_dir, f"{filename[0]}_input.png"))
            save_slice_as_png(target_normalized[0], 
                            os.path.join(target_dir, f"{filename[0]}_target.png"))
        
    # Stack into 3D volumes and save as NIfTI    
    if len(denoised_slices) > 0:
        denoised_volume = np.stack(denoised_slices, axis=0)
        print(f"Raw denoised volume shape: {denoised_volume.shape}")
        
        # Handle different possible formats
        if denoised_volume.ndim == 4:
            # If shape is [num_slices, channels, height, width]
            if denoised_volume.shape[1] == 3:
                denoised_volume = denoised_volume[:, 0, :, :]  # Take first channel
            # If shape is [num_slices, height, width, channels]
            elif denoised_volume.shape[-1] == 3:
                denoised_volume = denoised_volume[:, :, :, 0]  # Take first channel
        
        # Transpose to [height, width, num_slices]
        denoised_volume = np.transpose(denoised_volume, (1, 2, 0))
        denoised_volume = denoised_volume.astype(np.float32)
        
        nifti_img = nib.Nifti1Image(denoised_volume, affine=np.eye(4))
        nifti_path = os.path.join(output_dir, f'{patient_name}_denoised_volume.nii.gz')
        nib.save(nifti_img, nifti_path)

    if len(reconstructed_slices) > 0:
        recon_volume = np.stack(reconstructed_slices, axis=0)
        if recon_volume.ndim == 4:
            if recon_volume.shape[1] == 3:
                recon_volume = recon_volume[:, 0, :, :]
            elif recon_volume.shape[-1] == 3:
                recon_volume = recon_volume[:, :, :, 0]
        recon_volume = np.transpose(recon_volume, (1, 2, 0))
        recon_volume = recon_volume.astype(np.float32)
        
        nifti_img = nib.Nifti1Image(recon_volume, affine=np.eye(4))
        nifti_path = os.path.join(output_dir, f'{patient_name}_reconstructed_volume.nii.gz')
        nib.save(nifti_img, nifti_path)
        print(f"Saved reconstructed volume: {nifti_path}")

    # Save input and target volumes
    input_volume = np.stack(input_slices, axis=0)
    print(f"Raw input volume shape: {input_volume.shape}")

    # Handle the format - slices might be [H, W, C]
    if input_volume.ndim == 4:
        # If shape is [num_slices, channels, height, width]
        if input_volume.shape[1] == 3:
            input_volume = input_volume[:, 0, :, :]  # [155, 256, 256]
        # If shape is [num_slices, height, width, channels] 
        elif input_volume.shape[-1] == 3:
            input_volume = input_volume[:, :, :, 0]  # [155, 256, 256]
    elif input_volume.ndim == 3 and input_volume.shape[1] == 3:
        # If shape is [155, 3, 256] we need to reshape properly
        input_volume = input_slices[0].cpu().numpy() if torch.is_tensor(input_slices[0]) else input_slices[0]
        # Get the actual shape
        if input_volume.shape[0] == 3:  # [C, H, W]
            input_volume = np.stack([s[0, :, :] if s.shape[0] == 3 else s[:, :, 0] for s in input_slices], axis=0)
        else:  # [H, W, C]
            input_volume = np.stack([s[:, :, 0] for s in input_slices], axis=0)

    # Now transpose to nifti format [H, W, slices]
    if input_volume.ndim == 3:
        input_volume = np.transpose(input_volume, (1, 2, 0))
        
    input_volume = input_volume.astype(np.float32)
    nifti_img = nib.Nifti1Image(input_volume, affine=np.eye(4))
    nib.save(nifti_img, os.path.join(output_dir, f'{patient_name}_input_volume.nii.gz'))

    # Same for target volume
    target_volume = np.stack(target_slices, axis=0)

    if target_volume.ndim == 4:
        if target_volume.shape[1] == 3:
            target_volume = target_volume[:, 0, :, :]
        elif target_volume.shape[-1] == 3:
            target_volume = target_volume[:, :, :, 0]
    elif target_volume.ndim == 3 and target_volume.shape[1] == 3:
        target_volume = np.stack([s[0, :, :] if s.shape[0] == 3 else s[:, :, 0] for s in target_slices], axis=0)

    if target_volume.ndim == 3:
        target_volume = np.transpose(target_volume, (1, 2, 0))
        
    target_volume = target_volume.astype(np.float32)
    nifti_img = nib.Nifti1Image(target_volume, affine=np.eye(4))
    nib.save(nifti_img, os.path.join(output_dir, f'{patient_name}_target_volume.nii.gz'))
        
    return {
        'denoised': denoised_volume if len(denoised_slices) > 0 else None,
        'reconstructed': recon_volume if len(reconstructed_slices) > 0 else None,
        'input': input_volume,
        'target': target_volume
    }


def main(
    device_idx=0,
    ckpt="./checkpoints/epoch=000833.ckpt",
    config="./configs/sd-brats-finetune-c_concat-256.yaml",
    data_path="./ASNR-MICCAI-BraTS2023-GLI/",
    output_dir="./output_volumes",
    batch_size=1,
    ddim_steps=50,
    guidance_scale=7.5,
    ddim_eta=0.0,
    image_size=256,
    save_individual_slices=True,
    generate_denoised=True,
    generate_reconstruction=False
):
    """
    Main inference function
    
    Args:
        device_idx: GPU device index
        ckpt: Path to model checkpoint
        config: Path to model config
        data_path: Path to BraTS dataset
        output_dir: Directory to save outputs
        batch_size: Batch size (typically 1 for inference)
        ddim_steps: Number of denoising steps (more = better quality but slower)
        guidance_scale: Classifier-free guidance scale (higher = more faithful to condition)
        ddim_eta: Stochasticity in sampling (0 = deterministic DDIM)
        image_size: Image resolution
        save_individual_slices: Whether to save individual PNG slices
        generate_denoised: Whether to run full denoising process
        generate_reconstruction: Whether to generate encode-decode reconstruction
    """
    
    # Setup device
    device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    config_obj = OmegaConf.load(config)
    model = load_model_from_config(config_obj, ckpt, device=device)
    print("Model loaded successfully")
    
    # Setup data loader
    print("Setting up data loader...")

    # Create test configuration
    test_config = {
        'validation': False,
        'image_transforms': {
            'size': image_size
        }
    }

    dataset = BratsDatasetModuleFromConfig(
        root_dir=data_path, 
        batch_size=batch_size, 
        total_view=1,
        test=test_config, 
        num_workers=1
    )

    dataset.test_paths = dataset.train_paths + dataset.val_paths  # Use all data for inference

    dataloader = dataset.test_dataloader()
    # Setup sampler
    sampler = DDIMSampler(model)

    #To ensure device consistency
    original_apply_model = model.apply_model

    def apply_model_with_device_fix(x_noisy, t, cond):
        # Get the device of the model 
        model_device = x_noisy.device
        
        # Ensure time tensor is on correct device
        if torch.is_tensor(t):
            t = t.to(model_device)
        
        # Fix conditioning tensors
        if isinstance(cond, dict):
            fixed_cond = {}
            for key, value in cond.items():
                if isinstance(value, list):
                    fixed_cond[key] = [v.to(model_device) if torch.is_tensor(v) else v for v in value]
                elif torch.is_tensor(value):
                    fixed_cond[key] = value.to(model_device)
                else:
                    fixed_cond[key] = value
        else:
            fixed_cond = cond
        
        return original_apply_model(x_noisy, t, fixed_cond)

    model.apply_model = apply_model_with_device_fix
    
    # Process all slices and generate 3D volumes
    print(f"\nStarting inference with {ddim_steps} denoising steps...")
    
    volumes = process_batch_to_3d_volume(
        dataloader=dataloader,
        model=model,
        sampler=sampler,
        output_dir=output_dir,
        h=image_size,
        w=image_size,
        ddim_steps=ddim_steps,
        scale=guidance_scale,
        ddim_eta=ddim_eta,
        device=device,
        save_individual_slices=save_individual_slices,
        generate_denoised=generate_denoised,
        generate_reconstruction=generate_reconstruction
    )
    
    print("\n" + "="*50)
    print(f"Outputs saved to: {output_dir}")
    
    return volumes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BraTS Diffusion Model Inference - Generate 3D Volumes')
    
    # Model arguments
    parser.add_argument('--device_idx', type=int, default=0,
                        help='GPU device index (default: 0)')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/epoch=000833.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='./configs/sd-brats-finetune-c_concat-256.yaml',
                        help='Path to model config')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./ASNR-MICCAI-BraTS2023-GLI/',
                        help='Path to BraTS dataset')
    parser.add_argument('--output_dir', type=str, default='./output_volumes',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    
    # Sampling arguments
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='Number of denoising steps (default: 50, range: 25-200)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale (default: 7.5, range: 3.0-15.0)')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                        help='Stochasticity in sampling, 0=deterministic (default: 0.0)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image resolution')
    
    # Output arguments
    parser.add_argument('--save_individual_slices', action='store_true', default=True,
                        help='Save individual PNG slices (default: True)')
    parser.add_argument('--no_save_individual_slices', dest='save_individual_slices', 
                        action='store_false',
                        help='Do not save individual PNG slices')
    parser.add_argument('--generate_denoised', action='store_true', default=True,
                        help='Run full denoising process (default: True)')
    parser.add_argument('--no_generate_denoised', dest='generate_denoised', 
                        action='store_false',
                        help='Skip denoising process')
    parser.add_argument('--generate_reconstruction', action='store_true', default=False,
                        help='Generate encode-decode reconstruction (default: False)')
    
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    volumes = main(
        device_idx=args.device_idx,
        ckpt=args.ckpt,
        config=args.config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        ddim_eta=args.ddim_eta,
        image_size=args.image_size,
        save_individual_slices=args.save_individual_slices,
        generate_denoised=args.generate_denoised,
        generate_reconstruction=args.generate_reconstruction
    )
    
