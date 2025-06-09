# Finetuning & LoRAs
Low Rank Adapters

# Finetuning 
is the common practice pf taking a model which has been trained on a wide and diverse dataset, and then training it a bit more on the dataset you are specifically interested in.
This is a common practice on deep ;earning and has been shown to be tremendously effective all manner of models from standard image classification networks to GANs.

In this example we'll show how to fine tune Stable Diffucsion on a Pokemon dataset to create a text to image model which makes custom pokemon based on any text prompt. 

# LoRA

Before Flux loRAs were used to fix eyes and hands 
find LoRAs in Civit AI and Hugging Face

#### Usefull LoRAs
Detailed Tweaker : Eyes & Skin
Epi Noise Offset : for Potrait 
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

# Preparation 
3 Things important for FineTuning 
1. Image/ Dataset/Training data
     - 10 to 20 images of a person or object (for SD1.5 images should be 512x512, for XL models it can be 1024)
     - Use [Birme](https://www.birme.net/) for resizing Images depending on the model
2. Model
3. Caption/Labels
     - Blip captioner (is a vision model)
     - Wd tagger ( gives only tags. good for Anime)
     - interrogate Clip

# LoRA Training using KohyaSS

#### Step 0: Resize the image
- using Birme resize the image
- 512x512 for sd1.5 models
- 1024x1024 for XL models
- any size image will work with xl models 

#### Step 1 : Start a Kohya instance
- Start a new instance on JarvisLabs
- JarvisLabs --> Templates --> Kohya --> Run on Cloud
- use A6000 graphics

#### Step 2 : Upload images 
- open JupyterLab on instances.
- go to home/
- Create a new folder called "dataset"
- Upload all the sample images

![image](https://github.com/user-attachments/assets/ea38dbdf-b006-48f2-a721-4006a717a06e)

#### Step 3 : Watch the Logs
- Go to home. Open a "Terminal" from Launcher panel
- Copy path of Khoya_logs.log from left navigation. (watch you terminal path and provide relative path of the log)
```bash
#path copied from earlier step
tail -f kohya_logs.log
```

#### Step 4 : Generate Caption txt using Blip
- go to APIs
- go to Utilities --> Blip Captioning --> images path --> Click Caption Images (watch the Terminal for updates)
 
![image](https://github.com/user-attachments/assets/980124d7-67be-4d2e-897b-8a067d638d1f)

- See all the captions txt files created in dataset path

#### Step 5 : Add Tags Manually
- Manually add Tags/variables 

![image](https://github.com/user-attachments/assets/48a97c63-e5c3-4460-9284-a36718fdeb44)

- Add all fields values as shown below
- Add tags to every image
![image](https://github.com/user-attachments/assets/0fa5ed4c-782d-4c47-83fd-e0b1aee430bb)

#### Step 6 : Dataset Preparation
- Create a model training folder structure
- go to Lora tab and Dataset Preparation
![image](https://github.com/user-attachments/assets/1fab3cd5-f52c-446a-8862-73974f12ac4e)

- Add all fields values as shown below. Including Destination training directory
![image](https://github.com/user-attachments/assets/e003597d-9e12-4349-aee6-53d85d38eadb)

- Click "Copy info to respective fields"

- Steps/Cycle is called epoch 
- For good results in SD1.5 loRA model you will train for 2000-3000 step.
- "repeat" means: How many times to repeat the dataset in 1 training cycle. 
- "Batch Size": is a demominator in the epoch formulae
- (sample images/batch size) * Repeats = steps/cycle or Epoch
- Chose your "repeats" based on the "batch size" which you will choose in "paraments" tab.
- batch size of 2 is ideal, but 1 works as well
  
![image](https://github.com/user-attachments/assets/decc1f06-6636-4a9c-bd41-1832eaa32e8a)



#### Step 7 : Choose a Model
- Go to models- select a model
- Choose "bf16" (bcos we have a better graphics avaliable in A6000 instance)
![image](https://github.com/user-attachments/assets/6e2dde03-24c3-4de1-aa23-1a42acc54277)


#### Step 8 : Configure Acclerate Tab
- go to Acclerate model, change to "bf16" same as "Model tab" 
![image](https://github.com/user-attachments/assets/133a26b1-ed3e-4f27-af13-f2d4bcadd57e)

#### Step 9 : Configure Parameters Tab
- go to "parameters"
- Keep LoRA type as standard. as it works the best
- Note: there are 30 sample images in my dataset
- Earlier we had given "Repeats" as 10. which is like 300 steps/cycle (ie 30 images * 10 repeats = 300 steps/cycle). To get between 2000-3000 steps, we need 10 such cycles. Therefore Epoch or Steps/Cycle is 10.

![image](https://github.com/user-attachments/assets/d5ce193d-2881-4b8b-b387-94c8fc248249)

- we can use max training epoch or max train steps, epoch is better
- Explore with LR Scheduler
- Learning rate. default 0.00001. lesser the value more details it captures. precision of learning. below 0.00001 it will take forever to learn and we dont need that precision. value set at 0.00005 for learning faces.
![image](https://github.com/user-attachments/assets/316e4a78-6270-4362-b83b-ac7684f6f8bd)

- Network rank (Dimension): how much information will my LoRA save. if its face not much needed. for face keep the value between 16 - 32. for Style set to 128.
- Network Alpha - how much original model has to be distintive. keep this below Network rank, ideal to keep it half of Network rank

![image](https://github.com/user-attachments/assets/e44a3858-b97a-46eb-83ce-235aa606a36f)

#### Step 10
- Generate output during the process. Ideal to keep it at end of every cycle.

![image](https://github.com/user-attachments/assets/bc524d09-04f6-43d1-bd29-067ad2528a54)

#### Step 11 : Start Training 
- Start Training
- Check the logs in Jupyter labs

#### Step 12 : Verify and Download LoRA
- In Jupyter Labs go to home/training/model find the models generated. in our case there will be 10 models.
  ![image](https://github.com/user-attachments/assets/ec82ab63-71d2-43b7-b006-b75df33f5ed4)

- Go to home/training/model/sample and see the outputs. check at what epoch the output was good. say if the output was good at 3 epoch/900 steps, then under models download and use 3rd model.

![image](https://github.com/user-attachments/assets/cb475dfa-e77c-48f5-9156-d784201736b9)


#### Step 13 : Using LoRA
- Go to ComfyUI Instance --> JuptyerLabs
- upload the model to home/ComfyUI/Models/loras
- Note: watch for the progress bar at the bottom of the screen. if used the model before completing you will get "Incomplete buffer" error

   
![image](https://github.com/user-attachments/assets/c7262e3b-566f-4111-995a-e7202e96b8f8)

- Go to ComfyUI instance --> APIs
- Create a workflow and add a node called "Load LoRA" and for "lora_name" choose our new lora

![image](https://github.com/user-attachments/assets/5726daeb-e047-4269-b6ad-d7cbfeb997e7)

# FAQ
1. LoRA can be trained on Civit.ai models as well. Download the model on JupyterLabs and choose that model under "Model" tab.
2. Training a lora will take around 60 mins. check progress in Kohya_logs.log
   ![image](https://github.com/user-attachments/assets/eccea5b6-8caa-40c9-9340-ef95f2cdbaf7)
3. Can we train multiple lora one for face and one for Style on the same model checkpoint? Yes you can by connecting it in series in comfyUI. Believe the question was on using and not training.  





 



   
