# using Axolotl

in this example we will train LoRA. i.e. take a percentage of the base model retrain it and add a layer on it.  

---
#### Step 1:  
Pick a Axolotl template on Jarvislabs.
A5000pro is decent 
Increase Size to 200 GB 
<img width="1017" height="370" alt="image" src="https://github.com/user-attachments/assets/8494a590-daa0-489c-91d0-416a00248232" />

Open Jupiterlabs for the Instance 

goto--> /axolotl/examples to see avaiable models 

---
#### Step 2:
open instruct....yaml file. 
<img width="603" height="452" alt="image" src="https://github.com/user-attachments/assets/93285a97-b84a-4de0-af7b-b70c3380df4e" />

**Base Model**
<img width="647" height="192" alt="image" src="https://github.com/user-attachments/assets/a4d34cd4-568e-407d-8bb9-60c71abe079e" />

**Datasets**
<img width="713" height="123" alt="image" src="https://github.com/user-attachments/assets/b67bb0fe-b7a7-42ef-9243-431bb8f9d5d8" />

**Context Length**
<img width="627" height="103" alt="image" src="https://github.com/user-attachments/assets/68bb60d4-4646-45bf-827f-6e7e46da7fd9" />

**Lora**
Keep them standard. or can be reduced or increased.
you can chnage type to QLora and change settings 
<img width="598" height="201" alt="image" src="https://github.com/user-attachments/assets/e9695b56-f1ef-4efd-94df-17544d568284" />

**Monitoring ML tools**
ML training tools like Logs, GPU memory usage. Not must have. 
No chnages needs. Not must have. 
<img width="483" height="103" alt="image" src="https://github.com/user-attachments/assets/b271b10c-8230-4d91-bfbb-b8614988c492" />

**Imp settings**
Keep it to default or chnage only required.
<img width="498" height="123" alt="image" src="https://github.com/user-attachments/assets/8e0f6480-875a-4b25-a33a-0ec49fa05753" />

**Warm up**
Deepspeed: needs a diff package. Makes training faster and optimized. in trial state
<img width="494" height="242" alt="image" src="https://github.com/user-attachments/assets/3591a3f2-dff8-45b7-8d8b-73ffe9f0c7e7" />

---
#### Step 3:
Run the commands as provided in https://docs.axolotl.ai/docs/getting-started.html
The output should look like https://colab.research.google.com/drive/1Xu0BrCB7IShwSWKVcfAfhehwjDrDMH5m

---
#### Step 4:
Understanding the output:
- Tokenizer
- Dataloading 
- Loss: Loss functions. How much new your model is learning. how much lose the answers as from truth, ie how much the model already know. If from beggning these values are low, then the model already knows what its learning. If loss value is big and during trainings its recuding gradually, them your model is learning & hyperparameters are well tuned. Its common to see values fluctuating. but when you see large fluciations after a while in training, your learning rate is high.  
its value might not go to zero.
if there is no loss at all, then training value is incorrect or hyperparameters are incorrect. 
- Learning rate:
- Epoch

Success of Eval is not very dependable, as the sample for evel itself is very low 2-5%.

<img width="1335" height="423" alt="image" src="https://github.com/user-attachments/assets/16fdceca-e1c3-4920-af0c-1c6f30d406c1" />

#### Step 5:
Run merge LoRA weights commands. 

#### Step 6:
Optional to publish on Hugging face.

# Llamafactory
its a GUI based tool for training, eval. Requires Windows machine but slowing moving to Linux.  
[github](https://github.com/hiyouga/LLaMA-Factory)

# Other GPU providers
- runpod
- lambda.ai
- vast.ai
- fal.ai

# Run inference on **Replicate**
- Replicate has inbuild queue
- it has in build webhook
- cog.run is the container 

Follow the instructions 
[Replicate push-a-transformers-model](https://replicate.com/docs/guides/push-a-transformers-model)

cog.yaml - Configuration file 
predict.py - def setup: Change T5ForConditionalGeneration as suggested in the guide 
                        change Tokenizer to specific or leave it auto tokenizer
             def predict: max_length can be changed or default to 50 
More than one model can be run on the same device. Configure def setup.

# other inference providers
- AWS Sagemaker
  - [Deploy models for real-time inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html)
  - [deploy-llama2-7b-recording.ipynb](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Llama2/Llama2-7b/LMI/deploy-llama2-7b-recording.ipynb)
  - [Youtube](https://www.youtube.com/watch?v=UQWjKQe97Ew)
- replicate.ai
- modal.com
- fal.ai
            










