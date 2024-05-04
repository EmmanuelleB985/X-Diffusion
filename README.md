# X-Diffusion: Generating Detailed 3D MRI Volumes From a Single Image Using Cross-Sectional Diffusion Models

##  Usage
```
conda create -n XDiffusion python=3.9
conda activate XDiffusion
cd XDiffusion/scripts


We follow the installation instructions from [Zero-123](https://github.com/cvlab-columbia/zero123)


```
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

Download the Zero-123 checkpoint trained on Objaverse using the following source and place under folder Zero123:

```
https://huggingface.co/cvlab/zero123-weights/tree/main
wget https://cv.cs.columbia.edu/zero123/assets/300000.ckpt    
```

### Training

Run training command:  
```
python main.py \
    -t \
    --base configs/sd-brats-finetune-c_concat-256.yaml \
    --gpus 0 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from Zero123/300000.ckpt 
```

For inference:
```
python inference.py
```

Note that this uses around 30 GB of VRAM.


### Dataset (BRATS and UKBiobank)

BRATS2023 dataset can be downloaded by creating an account on https://www.synapse.org/#!Synapse:syn27046444/wiki/616571

UKBiobank can be downloaded after creating an account and registering on the UKBiobank platform. 
Follow instructions from this repo: 
https://github.com/rwindsor1/UKBiobankDXAMRIPreprocessing


##  Acknowledgement
This repository is based on [Zero-123](https://github.com/cvlab-columbia/zero123),[Stable Diffusion](https://github.com/CompVis/stable-diffusion). We would like to thank the authors of the abovementioned work for publicly releasing their code. 
