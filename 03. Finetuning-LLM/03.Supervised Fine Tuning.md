# Fine Tuning LLMs

## Data Preparation for LLM Fine tuning - Recap

### Goal:

Create a clean, diverse, high-quality dataset of **instruction–input–output** examples that your model can learn from effectively.

---

### The 7-Step Recipe

1. **Collect Raw Data**
    
    Gather domain-specific sources (e.g., meeting transcripts, emails, docs). Start messy - that’s normal.
    
2. **Clean & Preprocess**
    
    Remove filler words, fix speaker tags, standardize formats, redact PII, and normalize punctuation.
    
3. **Deduplicate & Decontaminate**
    
    Remove near-duplicate examples and anything overlapping with eval/test sets (if applicable).
    
4. **Annotate Instructions & Outputs**
    
    Convert raw input into instruction–output pairs using manual annotation or LLM-based generation (e.g., summarize, extract actions).
    
5. **Add Diversity & Complexity (Optionally augment data)**
    
    Vary task types, tones, lengths, and difficulty to ensure the model learns broadly. Use tools like **Augmentoolkit** or **Distilabel** to generate paraphrases, harder examples, or new tasks.
    
6. **Evaluate Quality**
    
    Run checks (automated + manual) to catch missing fields, inconsistent tone, or poor outputs. Optionally use LLM-as-a-judge for scoring.
    
7. **Iterate**
    
    Fine-tune a small model, find weak points, expand or fix dataset, and repeat until results stabilize.
    

---

### Key Takeaways

- **The model can only learn what your data teaches it.**
- **Good fine-tuning ≠ more data - it’s better data.**
- Most real-world effort is in **cleaning, annotation, and quality evaluation**, not just volume.

**Example: Instagram Script Generation**

```json
[
		{
        "instruction": "Write a short video script discussing a new AI tool that can convert screenshots of websites into working HTML and CSS code, in the style of the provided transcripts.",
        "input": "Article about a new open source GitHub repo by Avi Raja that uses ChatGPT-4, DALL-E 3 and computer vision to turn screenshots of websites into working HTML and Tailwind CSS code in seconds. Users can then interact with the generated code using natural language to modify elements like changing button colors.",
        "output": "You can now turn screenshots of any website into fully working code. And the best part, after generating code, you can interact with it like you're talking to a normal human being. You can say something like, turn this button into red or fix the colors in this website. This is an open source GitHub repo made by this guy called Avi Raja using ChatGPT-4, Vision, and DALL·E-3. You simply take a screenshot of the website that you want to clone, you put it in the tool, and that's about it. The tool generates HTML and Tailwind CSS for you in just seconds. In order to use this, you'll need to have access to the ChatGPT-4 Vision API case, and I believe this is going to be revolutionary tech. I've put a link to the tool down below. Go check it out and follow 100x."
    },
    {
        "instruction": "Create a humorous video script narrating a developer's daily routine in the style of a nature documentary, based on the provided transcripts.",
        "input": "Twitter video by Charlie Holt that uses GPT vision to interpret images from a webcam pointed at a developer every 5 seconds, and generates humorous narration in the style of David Attenborough using GPT-3 and 11Labs voice synthesis.",
        "output": "What's this? This is so funny. He's wearing what appears to be a blue fabric covering, which can only be assumed to be part of his mating display. That's David Attenborough giving an at-geo style commentary on a developer. Charlie Holt combined GPT vision and 11Labs using a Python script to create this fun video. He used vision to interpret images taken from his webcam every five seconds. GPT then interprets it in the style of Attenborough, and then 11Labs converts the script into his voice. The future of development is going to look something like this. Stitching together existing tools to create something completely new and novel. This is going to be the era of Gen AI engineers. Do you want to become one? Follow 100X."
    },
    {
        "instruction": "Write a short news update video script about Sam Altman returning as CEO of OpenAI, in the style of the example transcripts.",
        "input": "Sam Altman has returned as CEO of OpenAI, just 5 days after being fired, but now with a new board of directors. He was initially invited to Microsoft by Satya Nadella. Only one previous external board member, Adam D'Angelo, remains. Altman stated he is still committed to OpenAI's partnership with Microsoft.",
        "output": "Sam is back at OpenAI. It's almost like the scene from Wolf of Wall Street. He was initially invited by Satya Nadella to Microsoft, and the news broke out today that he's indeed making a comeback as the CEO of OpenAI. But this time, with a new board of directors. Only one out of the three external board members of the previous board survived this, Adam Angelo. He stated that he's still committed to their partnership with Microsoft. God, what a crazy story. He got fired and is back in only five days."
    }
 ]
```

## Hyperparameters

Hyperparameters are the settings you choose **before training** a model. They guide the learning process - like knobs you turn to control how a model learns.

**Real-world analogy:**

Think of fine-tuning like **training a chef** to cook a specific dish (say, Instagram-worthy vegan meals). Hyperparameters are the instructions you give:

- how long to train them (training steps)
- how fast they should learn (learning rate)
- how big their memory is (context length)
- how often they practice (batch size)

### 1. **`learning_rate`**

This controls **how fast the model updates** its internal understanding during training.

<img width="1436" height="431" alt="image" src="https://github.com/user-attachments/assets/3376aac7-9ac7-4a05-9465-d715f4eb50dc" />


### In simple terms:

If you're training the model to understand your brand’s Instagram voice - should it make **big leaps** after seeing each example (high learning rate), or **small, careful adjustments** (low learning rate)?

### Technical Insight:

- A too-high learning rate = unstable training, model forgets general knowledge
    - A too-low learning rate = painfully slow learning, may not capture your tone

---

### 2. **`num_train_epochs`**

Controls **how long you train** the model on your dataset.

### In simple terms:

How many times should the model go through your training examples (say 500 Instagram scripts)?

- 1 epoch = one full pass through the data
- More epochs/steps = more memorization, but also higher risk of overfitting

### Technical Insight:

- For small datasets, overfitting happens quickly - 2–3 epochs is often enough
- With LoRA, training is cheap, but excessive training = model copies examples rather than generalizing style

---

### 3. **`batch_size`**

How many samples the model sees **at once** before updating its weights.

### In simple terms:

Is the model learning from **one script at a time**, or **a group of scripts** in each mini-lesson?

- Larger batch size = more stable updates
- Smaller batch size = noisy learning, but useful if memory is tight

### Technical Insight:

- LoRA uses little memory, so batch sizes of `8`, `16`, or `32` are common

---

### 4. **`lora_r`** and **`lora_alpha`** (specific to LoRA)

These are **LoRA-specific hyperparameters** that control how the LoRA adapters work.

### `lora_r`:

- The **rank** of the low-rank matrix used to inject learnable parameters
- Think of it as **how much capacity** you're giving the LoRA module to learn
- Typical values: `4`, `8`, or `16`

### `lora_alpha`:

- A scaling factor — how strongly LoRA updates are applied
- Usually set to a multiple of `lora_r`, like `lora_alpha = 16` if `lora_r = 8`

---

### 5. **`cutoff_len` / `max_seq_length` / `context_length`**

This defines **how long your input and output can be**, in tokens.

**In the Instagram use case:**

You might be passing the brand brief + tone + 3 example scripts as input. If your max length is too low, parts will get **cut off** and the model won’t learn properly.

- Common values: `512`, `1024`, or `2048`

---

## Typical Hyperparameter Values

**For LoRA/QLoRA fine-tuning on 24–48GB VRAM, 500 data samples**

| Hyperparameter | Typical Value | Notes |
| --- | --- | --- |
| `learning_rate` | `2e-5` to `5e-5` | Lower for stable learning; QLoRA can handle slightly higher |
| `num_train_epochs` | `3–10` | With 500 samples, 3–5 is often enough to generalize without overfitting |
| `batch_size` | `4–8` | Memory dependent; larger batch = faster but requires gradient accumulation |
| `max_seq_length` | `512` or `1024` | Depends on input format (prompt + context + response) |
| `warmup_steps` | `30–100` | Helps prevent instability early on |
| `lora_r` | `8` | More than enough for most cases |
| `lora_alpha` | `16` or `32` | Usually 2–4x the value of `lora_r` |

### Factors That Influence These Values

| Factor | How It Affects Hyperparameters |
| --- | --- |
| **Dataset Size** (500 samples here) | Fewer samples → fewer epochs; smaller learning rate to avoid overfitting |
| **Task Complexity** (e.g., style transfer vs QA) | Creative tasks (like Instagram script generation) may need slightly more training steps and a higher context length |
| **Model Size** (7B vs 13B) | Larger models may need more warmup and smaller learning rates |
| **VRAM/GPU constraints** | Smaller batches and more gradient accumulation on 24GB vs 48GB cards |
| **Response Length** | Longer outputs → increase `max_seq_length` (to 1024 or even 2048) |

---

## 🧰 **Best Tools for Fine-Tuning LLMs**

### 1. **Axolotl**

> A highly flexible and community-standard tool for fine-tuning LLMs using LoRA and QLoRA.
> 

### Why it’s great:

- Built on top of 🤗 Transformers + PEFT + DeepSpeed
- YAML-configurable → no need to write Python code
- Handles both local GPU and multi-GPU training
- Supports popular models: LLaMA, Mistral, Yi, etc.

### Ideal For:

- Anyone who wants **full control** over fine-tuning without reinventing scripts
- You want to test LoRA settings, load from Hugging Face, train on your own GPU or cloud

Links:

- https://github.com/axolotl-ai-cloud/axolotl
- [https://jarvislabs.ai/dashboard/templates/axolotl](https://jarvislabs.ai/dashboard/templates/axolotl)
- [https://colab.research.google.com/drive/1Xu0BrCB7IShwSWKVcfAfhehwjDrDMH5m](https://colab.research.google.com/drive/1Xu0BrCB7IShwSWKVcfAfhehwjDrDMH5m)
- [https://www.youtube.com/watch?v=HAYPoeC41fw](https://www.youtube.com/watch?v=HAYPoeC41fw)

### 2. **Llamafactory**

> A GUI-based interface for training, inference, and evaluation on LLaMA and similar models.
> 

### Why it’s great:

- Has a web interface for setting training configs
- Supports dataset loading, model training, inference — all in one place
- Good for beginners or quick experimentation

### Ideal For:

- Those who don’t want to touch command-line or YAML configs
- Educators or teams who want to quickly show the flow end-to-end

Links: 

- https://github.com/hiyouga/LLaMA-Factory

## **Before vs After Fine-Tuning**

---

### **Prompt Given to Both Models**

> Create a video script about Topaz Video AI for enhancing and transforming video footage, based on the style of the provided transcripts.

Topaz Video AI can stabilize, sharpen, slow down footage up to 16x, and upscale to 8K by generating new frames in microsecond gaps. Many film studios use this tool for capabilities impossible with traditional editing software.
> 

---

### **Before Fine-Tuning (Base Model Output)**

> “Are you tired of dealing with shaky, unsteady footage? Topaz Video AI's advanced stabilization tools make it possible to level out the most turbulent of shots, resulting in smooth, cinematic footage that wows audiences. Welcome to the world of Topaz Video AI, the revolutionary video editing software that's changing the game for creators, studios, and professionals alike.”
> 

**Why this is weak:**

- **Generic** intro with no real structure
- Feels like **stock marketing language**
- Doesn’t highlight any **specific features**
- Ignores the **tone/style** we’re targeting

---

### **After Fine-Tuning (LoRA-tuned Output)**

> “You can now turn your noisy videos into ultra-sharp videos using Video AI from Topaz. It can stabilize your footage, it can smoothen or sharpen your footage, it can slow down your footage up to 16 times regardless of the camera limits or the frame rate. That is an insane advancement which is not possible with existing video editing tools. The way it does this is by adding entirely new frames in these microsecond gaps in order to slow down the video. And you can also upscale your footage up to 8K. There are a bunch of film studios that are already using this tool. It's a paid tool, disclaimer. In a lot of ways, this tool is going to give a new superpower to filmmakers and editors. For more AI stuff, follow 100x.”
> 

**Why this is strong:**

- Clearly **structured** and **conversational** — feels like something you'd hear in a YouTube or IG script
- Includes **specific features** (slow-mo, upscaling, frame generation)
- Adds **casual human commentary**: “That is an insane advancement…”
- Includes a **disclaimer** and **call to action** — which aligns with your fine-tuning objective
