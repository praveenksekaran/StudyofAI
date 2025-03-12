# Control Net
1. [Basics](https://getimg.ai/guides/guide-to-controlnet)
2. [Advance](https://www.runcomfy.com/tutorials/mastering-controlnet-in-comfyui)
3. [Examples](https://comfyanonymous.github.io/ComfyUI_examples/controlnet/)

#### What is ControlNet?
ControlNet allows you to use your own pictures as a reference for the AI when creating new images. Think of it as giving the AI a visual blueprint in addition to your written instructions (the text prompt). It doesn't replace the main Text to Image model you select in the AI Generator; instead, they work in tandem.

Different ControlNet models are specifically trained on certain aspects of images, such as edges or poses. For instance, if you want the character you generate to be in a specific position, you can choose a model focused on poses. Here's how it works.

Additional models will first extract the desired pose from your source image. ControlNet will then use this pose information to guide the Text to Image AI. This way, the AI can generate an image according to your text prompt, but it will also incorporate the selected pose thanks to ControlNet.

#### Types of ControlNet models

#### Image Prompt (IP) Adapter

- 1 image input: Tries to re-create all the features of the image.
- 2 images input: tries to combine 2 images featurs 
![IP Adapter](https://github.com/user-attachments/assets/c49a6677-4d62-486d-b289-43a3ea975296)

- Takes image as prompt. Text prompts are not very relavent.
- if a face is provided as input, then its not same face you get. for Same face you use Indtand ID module

![ComfyUI_IPAdapter_Workflow](https://github.com/user-attachments/assets/2dd74890-b892-4f7a-9d1c-b75a2e9d9534)


#### InstandID 
- Is an upgraded version of IP adapter and its good for face. It maintains facial feature.
- Prompt is important
- To Install:
    - ComfyUI manager --> Nodes  --> Search for InstantID by Cubix. Install and restart comfy UI manager  
![instandID_Nodes](https://github.com/user-attachments/assets/2337e6ef-0307-4237-93fe-6d8127aca601)

    - Comfy UI Manager --> Model --> search instantID and install all onnx files along with others shown below and restart browser.
![instandID_Models](https://github.com/user-attachments/assets/0ca44c2f-d006-4e31-833f-62294ae464b5)

- use "apply InstandtID" with tag "InstantID"

#### Controlnet + IP adapter 
![CN+IPA](https://github.com/user-attachments/assets/e35b85e0-37db-4115-b485-af89e8cad5a6)

Outputs
![ComfyUI_Controlnet IPadapter_workflow2](https://github.com/user-attachments/assets/2397a2eb-bb34-4638-8306-8f92d588c0ce)

#### INpainting 
- inpating does not require controlnet or IP adapter
- - Inpainting can add prompt into an image
  - remove part of the image or replace (using instandID or IP Adapter)
  ![removeInpainting](https://github.com/user-attachments/assets/d9c483f9-7ce5-4153-8bd8-0f7fc9463883)

#### Virtual Trial 

![Virtual Trial](https://github.com/user-attachments/assets/9351d64f-f6a7-414d-87b7-64e63b79ca15)

#### Civit AI

- Install Civitai Comfy Nodes from Custom nodes on ComfyUI Manager. Restart & refresh browser
- Load "CivitAI Checkpoint Loader"
- get API ket from civitai.com settings
- get RealvisXL inpainting model [RealVisXL](https://civitai.com/models/139562?modelVersionId=297320)
- Use it as Checkpoint Loader

#### GroundingDino
- Manager --> Custom Node --> groundingDino --> Install --> Restart 
- Add Node called Grounding Dino SAMSegment (segment anything)
- add GroundigDinoModelLoader(segment anything) --> GroundingDINO_SwinB(938MB)
- add SAMModelLoader(segment anything) -->sam_vit_h(2.56GB)

![ComfyUI_GroundingDino_Controlnet_2](https://github.com/user-attachments/assets/5215f5d1-12be-4b00-bd71-39ab1c92ab9c)

#### Removing BG 
use BRIAAI Matting

![image](https://github.com/user-attachments/assets/e86b9b32-5fdb-473f-8cfb-9f49a2c8a0e6)


#### IC-Lighting
https://github.com/kijai/ComfyUI-IC-Light
https://github.com/lllyasviel/IC-Light

1. Create Gradient using spline ssplitter 
![image](https://github.com/user-attachments/assets/fea8060e-d9be-4e63-a318-02201e59b58a)


2. Create Light Source 
![image](https://github.com/user-attachments/assets/06f8d188-a1cb-46d8-ab28-a98b413421da)

3. Replace Background Image with Mask
![image](https://github.com/user-attachments/assets/c79741ff-2cbc-44b1-99bf-bbc1e8ea32ba)





  



