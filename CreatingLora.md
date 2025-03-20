# Finetuning & LoRAs
Low Rank Adapters

# Finetuning 
is the common practice pf taking a model which has been trained on a wide and diverse dataset, and then training it a bit more on the dataset you are specifically interested in.
This is a common practice on deep ;earning and has been shown to be tremendously effective all manner of models from standard image classification networks to GANs.

In this example we'll show how to fine tune Stable Diffucsion on a Pokemon dataset to create a text to image model which makes custom pokemon based on any text prompt. 

#LoRA

Before Flux loRAs were used to fix eyes and hands 
find LoRAs in Civit AI and Hugging Face

#### Usefull LoRAs
Detailed Tweaker : Eyes & Skin
Epi Noise Offset " for Potrait 
Better Portrait lighting : Portrait lighting

#### Unique about LoRA
- Before LoRA there was DreamBooth training. Its training the entire checkpoint.
- LoRA has few differences from DreamBooth that makes it espicially appealing as an alternative
  - Faster Training: Training new concepts on LoRA takes few mins
  - Smaller Outputs: Trained LoRA outputs are much smaller than DreamBooth outputs
  - Multiple concepts: You can combine multiple trained concepts in a single image (WIP)
  - Faster Image Generation: when you train your own DreamBooth model on Replicate, the model onlt stays warm when youu're actively usint it. With LoRA, youe're not running your own model, but rather running the one coleofsimo/Lora model, which always on and ready to server predictions.
  - Better at styles, worst at faces: Based on our exprementation, LoRA seems to do a better job at styles than DreamBooth, but favces are'nt as good. They are struch in uncanny valley, rather than looking precisely like. 


#### How LoRA work?

![LoRA Flow](https://github.com/user-attachments/assets/a1735745-1f28-441a-a7c6-886cc3a4bd41)

![image](https://github.com/user-attachments/assets/50bbe16f-f6e1-433c-baa7-56bf9a66407f)

![image](https://github.com/user-attachments/assets/3896b795-1d05-4bc7-b014-b229d10af610)

#### Tokens in FineTuning 
Role of token in FineTuning?
its a trigger word to invoke a model or LoRA. it a unique or rare string. its a variable. 
called variable, rare token, instance token 

#### Data Preparation 
3 Things important for FineTuning 
- Image/ Dataset/Training data
     - 10 to 20 images of a person or object (for SD1.5 images should be 512x512, for XL models it can be 1024)
     - Use [Birme](https://www.birme.net/) for resizing Images depending on the model
- Model
- Caption/Labels
     - Blip captioner (is a vision model)
     - Wd tagger ( gives only tags. good for Anime)
     - interrogate Clip

#### LoRA Training using KohyaSS
**Step 1**
- JarvisLabs --> Templates --> Kohya --> Run on Cloud
- use A6000 graphics
- Instances --> API

**Step 2**
open Jupyter on instances. go to home/

![image](https://github.com/user-attachments/assets/ea38dbdf-b006-48f2-a721-4006a717a06e)

- Create a new folder called "dataset"
- Upload all the sample images

**Step 3**
- Go to home. Open a "Terminal" from Launcher panel
- Copy path of Khoya_logs.log from left navigation
'''bash
#path copied from earlier 
tail -f kohya_logs.log
'''

**Step 4**
- go to APIs
- go to Utilities --> Blip --> images path --> Click Caption Images (watch the Terminal for updates)
 
![image](https://github.com/user-attachments/assets/980124d7-67be-4d2e-897b-8a067d638d1f)

- See all the captions txt files created in dataset path

**Step 5**
- Manually add Tags/variables 

![image](https://github.com/user-attachments/assets/48a97c63-e5c3-4460-9284-a36718fdeb44)

- Add tags to every image
![image](https://github.com/user-attachments/assets/0fa5ed4c-782d-4c47-83fd-e0b1aee430bb)

**Step 6**
- Create a model training folder structure
- go to Lora tab and Dataset Preparation
![image](https://github.com/user-attachments/assets/1fab3cd5-f52c-446a-8862-73974f12ac4e)

- Steps/Cycle is called epoch 
- to understand "repeats", understand this : to train a SD1.5 loRA model you will train for 2000-3000 step.
- "repeat" means: How many times to repeat the dataset in 1 training cycle. (note there are 30 sample images in my dataset)
- Understanding Batch Size: is a demominator on the sample images.
- (sample images/batch size) * Repeats = steps/cycle or Epoch
- Chose your "repeats" based on the "batch size" you will choose in "paraments" tab. 
  
![image](https://github.com/user-attachments/assets/decc1f06-6636-4a9c-bd41-1832eaa32e8a)

![image](https://github.com/user-attachments/assets/e003597d-9e12-4349-aee6-53d85d38eadb)

- Click "Copy info to respective fields"

**Step 7**
- Go to models- select a model 
![image](https://github.com/user-attachments/assets/6e2dde03-24c3-4de1-aa23-1a42acc54277)

**Step 8**
- go to Acclerate model, chnage to "bf16" same as "Model tab" 
![image](https://github.com/user-attachments/assets/133a26b1-ed3e-4f27-af13-f2d4bcadd57e)

**Step 9**
- go to "parameters"
- Earlier we had given "Repeats" as 10. which is like 300 steps/cycle (ie 30 images * 10 repeats = 300 steps/cycle). To get between 2000-3000 steps, we need 10 such cycles  therefore Epoch or Steps/Cycle is 10.

![image](https://github.com/user-attachments/assets/d5ce193d-2881-4b8b-b387-94c8fc248249)










   
