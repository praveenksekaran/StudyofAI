# Flux
is a family of diffusion/image models by [black forest labs](https://blackforestlabs.ai/announcing-black-forest-labs/). By people who were behind Stability AI (company that developed SDXL).
[Flux Example](https://comfyanonymous.github.io/ComfyUI_examples/flux/)

#### What is Flux?
is a cutting-edge AI image generator created by Black Forest Labs, the same team behind popular Stable Diffusion. its designed to push the boundaries of whats possible with Artificial intelligence in a visual creativity. At its core, FLUX.1 utilizes advanced algorithms to transform text prompts into highly detailed and visually stunning images. 

- Supports from 512 to 1500 px images 

# Model Variants
To strike a balance between accessibility and model capabilities, FLUX.1 comes in three variants: FLUX.1 pro, FLUX.1 dev and FLUX.1 schnell

#### 1. FLUX.1 pro
The best of FLUX.1, offering state-of-the-art performance image generation with top of the line prompt following, visual quality, image detail and output diversity.
- Its closed licensed model accessible via APIs only.
- Its ultimate tool for professional and big companies.
- Its great at following instructions and creating super-detailed high-quality images. 

#### 2. FLUX.1 dev
is an open-weight, guidance-distilled model for non-commercial applications. Directly distilled from FLUX.1 pro, FLUX.1 dev obtains similar quality and prompt adherence capabilities, while being more efficient than a standard model of the same size.
- Its perfect for playing around with ideas and testing creative concepts for non-commercial use.
- good for training LoRAs 

#### 3. FLUX.1 schnell
its the fastest model, is tailored for local development and personal use. FLUX.1 schnell is openly available under an Apache2.0 license.
- It takes 4 steps to generate images
- There are slight distortion but the output is better than SDXL.

#### Comparing Variants 

| Qualifier | FLUX.1 pro | FLUX.1 dev | FLUX.1 schnell |
| :---:         |     :---:      |    :---: |    :---: |
| Image Quality   | Best     | Better    | Good    |
| Speed   | Good     | Good    | Best    |
| Usability & Licensing   | Paid. Cloased. for Ent & Pros. high Quality images. Commercial purpose     | Non-commerical. for Student, Hobbyist & Researchers    | Open. Personal & Local development. rapid prototyping    |

# Flux workflow basics

- Load checkpoint node has 3 components i.e. MODEL, CLIP & VAE. For Flux workflow, these components has to be loaded seperately as induvidual nodes.
    - Load Diffusion Model node
    - DualClip Loader
    - Load VAE
 
 ![image](https://github.com/user-attachments/assets/a21466d7-3424-49ce-8e76-a5d20a10af5a)

 ![Normal_Flux_Workflow](https://github.com/user-attachments/assets/5f703016-0121-4f82-aedd-f3394cdcf28b)


#### Load Diffusion Model Node
- choose flux Dev as unet model. flux1-dev.safetensors
- Weight Type as fp8_e4m3fn_fast. weight type is the compression of the model associated with size. smaller model have distortions, but very very negligible
  
#### DualClip Loader Node
- connect both Positive and Negative clips to Dual Clip Loader
- Clip_name1: clip_i.safetensors
- clip_name2: t5xxl_fp16.safetensors
- type: flux
- Only positive clip is considered in Flux architecture
- Negative clip is required for ComfyUI architecture, but it does not make any difference to Flux WF. Flux ignores this  its suggested to keep empty.

#### Load VAE
- vae_name: ae_safetensors

#### Flux Guidance Node
- add strength to the prompt i.e. its addition or adding strength to CFG
- only added to Positive CLIP
- 3.5 is default. increasing this dosent produce good output

#### kSampler Node
- Make CFG to 1.0 as the strength is determined by Flux guiance node. 0 is ideal by usually kept at 1.
- Sampler_name dpmpp_2m
- Scheduler sgm_uniform
- steps 20

# Downloading the Lora/Models
- Got JupyterLabs of Instance
- Open Folder
  - for loras: /ComfyUI/models/loras/
  - for flux models : /ComfyUI/models/unet/
  - for style models : /ComfyUI/models/style_models/
- Open Terminal from left navigation
- goto Huggingface --> Profile --> Access Token --> Create new token  
- run the below command in the terminal

```bash
# replace Your_Huggingface_API_token with actual token
# find download url by following https://comfyanonymous.github.io/ComfyUI_examples/flux/
# the downloads are in hugging face. Get url for Loras or unet models and replace below 
wget --header="Authorization: Bearer Your_Huggingface_API_token" "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors"
```

huggingFace API Img
![image](https://github.com/user-attachments/assets/c88204e4-f968-4fa2-906c-4be7920834fd)

Download URL Img
![image](https://github.com/user-attachments/assets/157f1bb1-d88d-429b-9598-be2dae0378e0)


# Pix Conditioning (Controlnet) Flow
- There are only 2 models avaliable for flux; Canny and Depth
- Diffusion models are avalible for both these controls. However we can use distilled models (LoRA)/ i.e. use Flux1-dev in combination with flux1-canny-dev-lora.safetensors instead of flux1-canny-dev.safetensors.
- Download the models under /ComfyUI/models/loras/
- use node LoraLoaderModelOnly to load Depth or Canny
- Flux guidance should be increaded to 30 to increase the weight of the prompt

![Flux_Depth](https://github.com/user-attachments/assets/92c31436-fa76-48de-b0cd-9ed3993e1226)

# Fill (inpainting) Flow
- flux1-fill-dev.safetensors is avaliable as Unet model. hence download it under /ComfyUI/models/unet/
- InpaintingModelConditioning is the Node for inpainting

![Flux_Inpainting](https://github.com/user-attachments/assets/03444333-8f1b-4426-a861-aedce34a327f)

# Redux (IP Adapter) Flow
- Install sigclip_vision_3849(patch14_384) model in ComfyUI Manager or download sigclip_vision_patch14_384.safetensors to ComfyUI/models/clip_vision
- Download the complete model flux1-redux-dev.safetensors under /ComfyUI/models/style_models/
- Apply Style is the node
- Main model is Flux1-dev

  ![Flux_Redux_Workflow](https://github.com/user-attachments/assets/cf6c854f-4c8c-4362-b36b-cce0cb47fee0)


# FAQ
- In Redux flow prompt is not considered. Its becasue of the " Apply Style Model" was set at 1.0. if decreased, prompt would be considered.
- PullID is Equivalent if InstantID for Flux. PullID can also be used in SD workflow
- There is no IC Lighting Workflow with Flux.
- Use SamSegment along with Inpainting - Try 
- Normal/Basic Flux WF keep Guidance at 3.5, if used with any other flow keep guidance at 30. 
- Depth and Redux workflow is the killer workflow
- recraft.ai is better than flux and generates vector based. but its closed model.



      
