from contextlib import nullcontext
from functools import partial
import math
import fire
import gradio as gr
import numpy as np
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
from ldm.util import load_and_preprocess, instantiate_from_config
import os 
import random
import nibabel as nib
from ldm.data.bratsloader_test import BratsDatasetModuleFromConfig

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def compute_perceptual_similarity_from_list(pred_imgs_list, tgt_imgs_list,
                                            take_every_other,
                                            simple_format=True):

    # Load VGG16 for feature similarity
    vgg16 = PNet().to("cuda")
    vgg16.eval()
    vgg16.cuda()

    values_percsim = []
    values_ssim = []
    values_psnr = []
    equal_count = 0
    ambig_count = 0
    for i, tgt_img in enumerate(tqdm(tgt_imgs_list)):
        pred_imgs = pred_imgs_list[i]
        tgt_imgs = [tgt_img]
        assert len(tgt_imgs) == 1

        if type(pred_imgs) != list:
            pred_imgs = [pred_imgs]

        perc_sim = 10000
        ssim_sim = -10
        psnr_sim = -10
        assert len(pred_imgs)>0
        for p_img in pred_imgs:
            t_img = load_img(tgt_imgs[0])
            p_img = load_img(p_img, size=t_img.shape[2:])
            t_perc_sim = perceptual_sim(p_img, t_img, vgg16).item()
            perc_sim = min(perc_sim, t_perc_sim)

            ssim_sim = max(ssim_sim, ssim_metric(p_img, t_img).item())
            psnr_sim = max(psnr_sim, psnr(p_img, t_img).item())

        values_percsim += [perc_sim]
        values_ssim += [ssim_sim]
        if psnr_sim != np.float("inf"):
            values_psnr += [psnr_sim]
        else:
            if torch.allclose(p_img, t_img):
                equal_count += 1
                print("{} equal src and wrp images.".format(equal_count))
            else:
                ambig_count += 1
                print("{} ambiguous src and wrp images.".format(ambig_count))

    if take_every_other:
        n_valuespercsim = []
        n_valuesssim = []
        n_valuespsnr = []
        for i in range(0, len(values_percsim) // 2):
            n_valuespercsim += [
                min(values_percsim[2 * i], values_percsim[2 * i + 1])
            ]
            n_valuespsnr += [max(values_psnr[2 * i], values_psnr[2 * i + 1])]
            n_valuesssim += [max(values_ssim[2 * i], values_ssim[2 * i + 1])]

        values_percsim = n_valuespercsim
        values_ssim = n_valuesssim
        values_psnr = n_valuespsnr

    avg_percsim = np.mean(np.array(values_percsim))
    std_percsim = np.std(np.array(values_percsim))

    avg_psnr = np.mean(np.array(values_psnr))
    std_psnr = np.std(np.array(values_psnr))

    avg_ssim = np.mean(np.array(values_ssim))
    std_ssim = np.std(np.array(values_ssim))

    if simple_format:
        # just to make yaml formatting readable
        return {
            "Perceptual similarity": [float(avg_percsim), float(std_percsim)],
            "PSNR": [float(avg_psnr), float(std_psnr)],
            "SSIM": [float(avg_ssim), float(std_ssim)],
        }
    else:
        return {
            "Perceptual similarity": (avg_percsim, std_percsim),
            "PSNR": (avg_psnr, std_psnr),
            "SSIM": (avg_ssim, std_ssim),
        }

def get_inp(batch, k):
    x = batch[k]
    if len(x.shape) == 3:
        x = x[..., None]
    x = rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float()
    return x


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 20 * np.log10(255.0 / mse)
    return psnr


@torch.no_grad()
def run(self, batch,subroot, return_first_stage_outputs=False, force_c_encode=False,
                cond_key=None, return_original_cond=False, bs=None, uncond=0.05,device=None):
    i = 0

    for batch in d2.test_dataloader():
        target = batch["image_target"]
        inp = batch["image_cond"]
        filename = batch["filename"]
        T_cond = batch['T']
            
        precision_scope = nullcontext
        with precision_scope("cuda"):
            with model.ema_scope():
        
                x = inp
                if len(x.shape) == 3:
                    x = x[..., None]
                x = rearrange(x, 'b h w c -> b c h w')
                x = x.to(memory_format=torch.contiguous_format).float()
                
                T = T_cond.to(memory_format=torch.contiguous_format).float()
                #filename = batch['filename']
                
                if bs is not None:
                    x = x[:bs]
                    T = T[:bs].to(device)

                x = x.to(device)
                encoder_posterior = model.encode_first_stage(x)
                z = model.get_first_stage_encoding(encoder_posterior).detach()
                #cond_key = cond_key or self.cond_stage_key
                cond_key = "image_cond"
                xc = get_inp(batch, cond_key).to(device)
                if bs is not None:
                    xc = xc[:bs]
                cond = {}

                # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
                random = torch.rand(x.size(0), device=x.device)
                prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
                input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")
                null_prompt = model.get_learned_conditioning([""])

                # z.shape: [8, 4, 64, 64]; c.shape: [8, 1, 768]
                # print('=========== xc shape ===========', xc.shape)
                with torch.enable_grad():
                    clip_emb = model.get_learned_conditioning(xc).detach()
                    null_prompt = model.get_learned_conditioning([""]).detach()
                    cond["c_crossattn"] = [model.cc_projection(torch.cat([torch.where(prompt_mask, null_prompt, clip_emb), T[:, None, :]], dim=-1))]
                cond["c_concat"] = [input_mask * model.encode_first_stage((xc.to(device))).mode().detach()]
                out = [z, cond]
                if return_first_stage_outputs:
                    xrec = model.decode_first_stage(z)
                    out.extend([x, xrec])
                    
                    output_ims = []
                    count = 0
                    for x_sample in xrec:
    
                        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

                        img = torch.clamp((inp + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                        img = 255. * (img[0,:,:,:].cpu().numpy())
                        img = Image.fromarray(img.astype(np.uint8))
                      
                        tg = torch.clamp((target + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                        tg = 255. * (tg[0,:,:,:].cpu().numpy())
                        tg = Image.fromarray(tg.astype(np.uint8))
                                                
                        name_inp = "{}_sample_input_{}.png".format(filename[count],str(i))
                        name = "{}_sample_{}.png".format(filename[count],str(i))
                        name_tg = "{}_sample_tg_{}.png".format(filename[count],str(i))
                        
                        path = os.path.join(subroot, name)
                        Image.fromarray(x_sample.astype(np.uint8)).save(path)
                        
                        path_input = os.path.join(subroot, name_inp)
                        Image.fromarray(img.astype(np.uint8)).save(path_input)
                        
                        path_tg = os.path.join(subroot, name_tg)
                        Image.fromarray(tg.astype(np.uint8)).save(path_tg)
                        count +=1
        i += 1 

              
    if return_original_cond:
        out.append(xc)
    return out

@torch.no_grad()
def sample_model(input_im, cond, model, sampler, precision, h, w, ddim_steps, n_samples, scale, \
                 ddim_eta):
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            
            x = input_im
            if len(x.shape) == 3:
                x = x[..., None]
            x = rearrange(x, 'b h w c -> b c h w')
            x = x.to(memory_format=torch.contiguous_format).float()
   
            c = model.get_learned_conditioning(x).tile(n_samples,1,1)
            
            T = cond.to(memory_format=torch.contiguous_format).float()
            print("T",T)
    
            T = T[0][None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            print("C",c)
            cond = {}
            cond['c_crossattn'] = [c]
            c_concat = model.encode_first_stage((x.to(c.device))).mode().detach()
            cond['c_concat'] = [model.encode_first_stage((x.to(c.device))).mode().detach()\
                               .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None
            print("cond",cond)

            shape = [4, h // 8, w // 8]
            ddim_steps = 199
  
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
           
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


description = \
"""
Generate Novel View Synthetis given an Input Image using a Fine-Tuned version of Stable Diffision trained on Objeverse Dataset by Zero-123 (https://github.com/cvlab-columbia/zero123)
. Stable diffusion weights can be obtained from [Lambda](https://lambdalabs.com/),trained by [Justin Pinkney](https://www.justinpinkney.com) ([@Buntworthy](https://twitter.com/Buntworthy)).
__Get the [code](https://github.com/justinpinkney/stable-diffusion) and [model](https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned).__
![](https://raw.githubusercontent.com/justinpinkney/stable-diffusion/main/assets/im-vars-thin.jpg)
"""

article = \
"""
## How does this work?
The normal Stable Diffusion model is trained to be conditioned on text input. This version has had the original text encoder (from CLIP) removed, and replaced with
the CLIP _image_ encoder instead. So instead of generating images based a text input, images are generated to match CLIP's embedding of the image.
This creates images which have the same rough style and content, but different details, in particular the composition is generally quite different.
This is a totally different approach to the img2img script of the original Stable Diffusion and gives very different results.
The model was fine tuned on the [LAION aethetics v2 6+ dataset](https://laion.ai/blog/laion-aesthetics/) to accept the new conditioning.
Training was done on 4xA6000 GPUs on [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud).
More details on the method and training will come in a future blog post.
"""


if __name__ == "__main__":
    
    device_idx=0
    ckpt="./checkpoints/epoch=000833.ckpt"
    config="./configs/sd-brats-finetune-c_concat-256.yaml"
    
    path_data = "./ASNR-MICCAI-BraTS2023-GLI/"
    d2 = BratsDatasetModuleFromConfig(root_dir = path_data, batch_size = 1, total_view = 1, train=False)
    subroot = './'
    device = f"cuda:{device_idx}"
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)
    count = 0
    for batch in d2.test_dataloader():
        target = batch["image_target"]
        inp = batch["image_cond"]
        filename = batch["filename"]
        T_cond = batch['T']
        input_im = batch["image_cond"]
        input_im = (input_im).to(device)
        sampler = DDIMSampler(model)
        ddim_steps = 999
        run(d2, model.first_stage_key,subroot,
                                    return_first_stage_outputs=True,
                                    force_c_encode=True,
                                    return_original_cond=True,
                                    bs=1,device=device)