# Control Net
Basics : https://getimg.ai/guides/guide-to-controlnet
Advance: https://www.runcomfy.com/tutorials/mastering-controlnet-in-comfyui
Examples: https://comfyanonymous.github.io/ComfyUI_examples/controlnet/

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
  



