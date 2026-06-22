#  Introduction to Fine Tuning LLms

A model is made up of floating point numbers. The precision could be 8 bit....  32 bit etc. 

# When to use Fine-tuning 

#### Before fine-tuning 
- Start with prompt engineers
-  if the context is very large and tokens will increase cost
- Not-Tech reason: Enterprise value control and customizability of models

#### Fine-tuning for Enterprise 
  [Enterprise AI report](https://a16z.com/ai-enterprise-2025/)
- Fine-tuning viewed as less necessary as model capabilities improve
- There is a upfront cost in fine-tuning
- Companies with specific usecase and data will continue fine-tuning
- Cost, control and customizability are the considerations for Enterprise
- Ent Customization: 72% Fine-tuning, 22% RAG, 6% create custom model

#### Purpose of fine-tuning 
- Specializes a pre-trained model for specific tasks
- Optimizes performance on narrower, task-specific datasets
- Builds upon general lanaguage knowldge from pre-training 

# LLM Training Lifecycle

<img width="1070" height="411" alt="image" src="https://github.com/user-attachments/assets/c2ea61d5-bae9-4501-8cec-589b1ae1ec32" />

- Base Model 
Peta bytes  raw data is fed to neural networks, for the model to undersatnd the patterns, general language etc. Requires lot of compute power and time.
unsupervised learning.

#### Stages of training 
There are 3 stages 
- Pre-training: Uses raw text for next token prediction, resulting in a base model. unsupervised learning.
- Supervised Fine-tuning: uses dataset (e.g. question-answers pairs) to teach the model to follow instructions. Supervised learning.
- Preference Alignment: Aligns the model with human preference, resulting in a chat model.
  How humans needs output. bullet points, paragraph etc.

# Pre-training 

<img width="1151" height="510" alt="image" src="https://github.com/user-attachments/assets/e2fd7a31-66ad-4050-86e6-432c42c80dcd" />

#### Stage 1
- Data preparation : Prepare high quality dataset often terabytes
- Attention mechanism : An attention mechanism in machine learning is a technique that allows a model to focus on the most relevant parts of the input data when processing it, mimicking human cognitive ability to prioritize certain information. Instead of treating all input elements equally, attention assigns weights to different parts, emphasizing those that are most important for the task at hand. This selective focus improves the model's ability to understand complex relationships and dependencies within the data, particularly in tasks involving long sequences or high-dimensional data.
- LLM Architecture: The model's architecture, often based on the Transformer, allows it to process sequential data and capture long-range dependencies effectively
- Train a tokenizer for the data
- pre-process the dataset using the tokenizer 

#### Stage 2
- configure number of Training Loop --> Evaluate the output --> Load pretrained weights --> reconfigure training loops
- Model learns to predict words or fill in missing text
- Uses self-supervised learning techniques
- Two main approaches:
    - Masked Language Modeling: predict intentionally hidden tokens'
    - Casual Lanaguage Modeling: Predict the next word given preceding context
- Trains on vast amount of data to develop lanaguage understanding

#### Stage 3
- Model develops general lanaguage knowldge
- Becomes a proficient lanaguage encoder
- Lacks specific task or domain knowldge 
- Fine-tune with dataset. 
  - Classifier : if you fine-tune with Class label dataset
  - Personal assistant: if fine-tuned with instruction dataset
 
#### Supervised Fine-tuning 
Fine-tuning is a process where a pre-trained model, which has already learned some patters and features on a large dataset, is further trained (fine-tuned) on a smaller, domain-specific dataset. in the context of "LLM-fine tuning" LLM refers to Large Lanaguage Models like GPT series from OpenAI. This method is important because training a LLM from scratch is incredibly expensive, both in terms of computational resources and time. By leveraging the knowldge already captured in the pre-training model, one can achieve high performance on specific tasks with significantly less data and compute.

#### Transfer Leraning 
The primary advantage of transfer learning is that the model starts with a significant amount of pre-learned knowldge. instead of learning from scratch, the model uses its existing knowldge base to adapt more quickly and effectively to the new task. This leads to improved performance, reduced training time, and ofter requires less labeled data for the new task. 

#### Fine-tuning process 

<img width="1030" height="566" alt="image" src="https://github.com/user-attachments/assets/65524a07-7be8-4322-9630-a0ba2498f5f9" />

- Previsous learned parameters: e.g. precision bits
- loss function: How close to correct answer, gives small number. Measures how "off" the models output is from desired results. no loss means the model is not learning anything
- Parameter or hyper parameters

#### RLHF & Preference alignment 
Reinforcement Learning from Human Feedback (RLHF)
<img width="691" height="240" alt="image" src="https://github.com/user-attachments/assets/0d9ddfc3-9fe1-4f53-86e7-e54e13c7d64b" />


Proximal Policy Optimization (PPO) - 
Reward model - a model created on preferred data. model gives 2 answers, you select one answers called prefered answers. 

#### Core idea 
we have a smart large Language maodel but dosent know how to properly communicate to human, to over come 
- Make users(humans) interact with GPT3, show them alternative answers and collect their preserred one
- Gather the feedback in a dataset with the following shape:a. User question | preferred answer | rejected answer
- Train a reinforcement learning (RL with PPO) model dataset
- with RL model trained, make it evaluate GPT3 answers and give it rewards fine-tuning the model

#### Direct Preference Optimization (DPO)
<img width="824" height="410" alt="image" src="https://github.com/user-attachments/assets/e3627e12-be23-47d4-b2f8-3ba1f0ffc6a3" />

Training a reward model from humam feedback was complex and ofter unstable. Hence some thought that just skipping the reward model and direcly fine-tune an LLM with the preference dataset could work. and Suprise it worked.
The core idea of the stanford team was that, internally, LLMs itself worked as a reward model and just giving it preference data would do the trick. 

# Supervised Fine-Tuning Training Techniques 
<img width="899" height="458" alt="image" src="https://github.com/user-attachments/assets/2cc1c3e4-5f92-4232-b884-a3969b7b5c15" />

#### Full Fine-tuning 
- Updates all model parameters i.e. weights
- Requires significant computational resources
- Provides max adaptation to new tasks

#### LoRA
- Updates a small number of tasks-specific parameters
- Much more efficient than full fine-tuning
- Preserves most of the originals models knowldge
- Can be combined with the original model weights
- Frezes existing model weights and creates a new layer
- LoRA can be used in another model from the one it was trained on
- create a new file for LoRA

#### QLoRA 
- Combines LoRA with quantization techniques (4 bit precision). Quantization is reduction in precision 
- Even more memory-efficient that Standard LoRA
- Allows fine-tuning of larger models on consumer hardware
- Many have small tradeoff in performance compared to full-precision LoRA

# Hyperparameters

#### Learning Rate
The learning rate controls the size of updates to the model weights during training. A high rate allows for faster learning but can cause instability or convergence to a suboptimal solution., while a low learning rate can result in slow convergence or getting struck in local minima.

<img width="1016" height="304" alt="image" src="https://github.com/user-attachments/assets/a7ff302a-7cd6-417c-a4c5-96da2573e5af" />













