# Evaluating LLMs

## Why Evaluate?

> Why should we care about evaluating LLMs after fine-tuning?
We just fine-tuned a model to generate Instagram scripts. How do we know it actually got better?
> 

### 1. LLMs Are Unpredictable

Even top models can:

- Hallucinate facts
- Be overly vague or overly confident
- Struggle with formatting (especially important in Instagram captions!)

📌 Imagine our fine-tuned model starts generating captions that are funny but include made-up product names. It sounds good, but it’s useless for real-world use.

### 2. Just Looking at a Few Good Examples ≠ Confidence

- One or two good outputs don’t prove anything.
- It’s like saying your software works because you ran it once and it didn’t crash.

📌 We wouldn’t release a product based on 5 successful test runs. So we shouldn’t trust an LLM on 5 good responses either.

### 3. You Need Tests Just Like You Do With Code

- Evaluation = writing tests for language behaviour.
- We want to catch problems **before users do**.
- Helps us improve over time.

📌 Think of evaluation like building a QA checklist for our Instagram script generator.

### 4. Evaluation Isn’t Easy (Yet)

- In traditional ML, we had accuracy, precision, recall.
- But LLMs are fuzzy. Output can *look* right but be wrong in subtle ways.
- So we need new methods: human reviews, benchmarks, auto-evaluators, etc.

## Challenges with Evaluating LLMs

### 1. **Traditional ML is Easier to Measure**

In traditional ML:

- You have fixed inputs → fixed outputs
- Evaluation = clear metrics (accuracy, score, etc.)
- Example: “Is this spam or not?” → Binary → Easy to score

<img width="800" height="356" alt="image" src="https://github.com/user-attachments/assets/9fbe3a5d-91e7-4478-95ec-a69b5f72f83b" />


📌 **Contrast that with LLMs:**

- LLMs can generate multiple valid answers.
- You can't just say “correct/incorrect”.

> Let’s say we prompt our fine-tuned model to generate an Instagram script for a new headphone launch. There’s no one right answer - several variations could work.
> 

---

### 2. **LLM Evaluation Is Subjective + Fuzzy**

- Outputs vary in **style, tone, length, clarity, creativity, factuality**.
- Evaluating requires **judgement**, not just number-crunching.

📌 Some scripts may look good but exaggerate product features. Others may be factually correct but boring. Which one is better?

---

### 3. **Automated Metrics Fall Short**

- BLEU, ROUGE, etc. don’t capture creativity or intent well.
- Semantic similarity tools help but aren’t enough.
- LLMs can trick themselves if used as evaluators.

---

### 4. **Human Evaluation Isn’t Perfect Either**

- Humans are inconsistent
- Subject to bias (e.g. preferring long or funny answers)
- Expensive and time-consuming

📌 Imagine you ask 3 people to score your script outputs from 1–5. One loves dramatic captions. Another prefers punchy one-liners. Their 5s and 2s won’t match.

## Evaluation Methods
<img width="1912" height="874" alt="image 1" src="https://github.com/user-attachments/assets/b38a373f-141b-4612-b33f-77e3dfa728ba" />

### 1. Public Benchmarks

- These are popular research datasets used to measure how well base models like GPT, Claude, and LLaMA perform.
- Examples:
    - [**HumanEval**](https://paperswithcode.com/sota/code-generation-on-humaneval) → Code generation accuracy
    - [**HellaSwag**](https://paperswithcode.com/dataset/hellaswag) → Common sense & reasoning
    - [**MMLU**](https://paperswithcode.com/dataset/mmlu) → Standardized tests across subjects
    - [**TruthfulQA**](https://paperswithcode.com/dataset/truthfulqa) → How factual the answers are
    - [**ARC**](https://paperswithcode.com/dataset/arc) → Multiple choice reasoning (like science Olympiads)
    - [**Chatbot Arena**](https://lmarena.ai/) → Crowd-sourced model preference voting
- Example: [Meta Llama 3 Benchmarks](https://ai.meta.com/blog/meta-llama-3/)

**Pros:**

- Cheap, fast
- Standardized for comparison

**Cons:**

- Not built for your use case (e.g., Instagram scripts)
- Irrelevant formats and styles
- Doesn’t reflect your prompting or tone

📌 Benchmarks are great if you’re building the next GPT-5. But we’re just trying to make better Instagram scripts. These benchmarks won’t help us much.

### 2. Automated Evaluation using LLMs (AutoEval)

> Letting an LLM judge other LLM outputs.
> 

There are two main tools:

### **Auto-Evaluator**

- You define criteria: clarity, tone, engagement, brand fit, hallucination, etc.
- Use GPT-4o or Claude to **score** outputs of your fine-tuned model.
- You can automate grading across dozens or hundreds of examples.

📎 [Using LLM-as-a-judge](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/llm_judge.ipynb)

**Bias Warning:**

- LLMs tend to prefer longer outputs
- They might “agree with themselves”
- They can’t always justify their scores
- They might not handle style or nuance well

📌 You might ask GPT-4o to rate captions on a scale of 1–5, but it might give everything a 4.5. And it might prefer the longer caption just because it ‘feels’ more complete.

### 3. Human Evaluation

Still the **gold standard**, especially for creative tasks.

- You create a rubric:
    - Is it engaging?
    - Is it brand-safe?
    - Is it clear and concise?
    - Would you actually post it?
- Ask multiple reviewers to rate and compare outputs.
- Time-consuming but more aligned with **real-world judgement**.

**Pros:**

- Captures nuance, tone, vibe
- Great for subjective tasks like writing, design, branding

**Cons:**

- Expensive
- Slow
- Inconsistent if rubric is unclear

📌 We had 5 people from our marketing team score outputs from our fine-tuned model. The ones they liked the most weren’t always the most ‘correct’ - but they matched the brand’s tone.

## Recipe for Evaluating LLMs

<img width="883" height="357" alt="image 2" src="https://github.com/user-attachments/assets/193a6337-bb56-49e5-96d9-bf4df7fe881d" />

(Human-verified LLM Evaluation)

### 1. Developer Creates Gold Examples

Start with 5–10 high-quality prompt-output pairs that represent the *ideal output* for your use case.

📌 **Example**:

Prompt: “Announce launch of new noise-cancelling headphones”

Gold Output: “Silence everything. Hear only what matters….”

These gold examples are your baseline — they tell the auto-evaluator and humans what “great” looks like.

---

### 2. Generate More Test Prompts (Optional)

Use LLMs to simulate a wide range of realistic and edge-case prompts. This builds a rich **test suite**.

📌 Add variety:

- Promotional tones
- Emoji-heavy prompts
- Product feature deep-dives
- Influencer-style vs technical-style requests

---

### 3. Verified Auto-Eval (Middle of the Feedback Loop)

Use an LLM (like GPT-4) to score and review generated outputs **against your gold outputs and rubric**.

Tools:

- [Auto-Evaluator](https://github.com/rlancemartin/auto-evaluator) – takes in gold references, LLM outputs, and a rubric
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) – great for standardized tasks, less so for creative tasks

Tips:

- Ask LLMs to score on dimensions like:
    - Factual accuracy
    - Brand voice consistency
    - Creativity
    - Call-to-action clarity
- Always request **rationale** with the score - not just a number
- Refine prompts, formats, and scoring logic iteratively based on auto-eval behaviour.

---

### 4. High-Quality Human Evaluation (Right side of the loop)

Involve trusted evaluators who understand the use case.

📌 Provide a rubric:

| Criteria | Question |
| --- | --- |
| **Clarity** | Is the message easy to understand? |
| **Brand Fit** | Does it match our voice? |
| **Engagement** | Is it catchy/scroll-stopping? |
| **Factuality** | Any exaggerations or errors? |

Prefer **A/B comparisons** over abstract scores.

**Auto-eval ↔ Human eval**:

- Use human feedback to validate whether auto-eval scores make sense
- Refine your auto-eval instructions or gold data if they don’t align

---

### 5. Loop: Iterate and Expand

Back to the **developer**:

- Add more examples for weak spots
- Expand the rubric based on user feedback
- Use failures to improve prompts, fine-tuning data, or eval logic

This is **continuous evaluation** – not a one-time task.

## LLM as Judge
[Original article](https://huggingface.co/learn/cookbook/en/llm_judge)

### Automatic Evaluation Script: LLM as a Judge (Simplified)

```python
# Required Libraries
# pip install --upgrade openai pandas

import openai
import pandas as pd
import json
import time

# Set your API key
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual key

# -------------------------
# Expected JSON Format
# -------------------------
# [
#   {
#     "prompt": "Announce launch of new noise-cancelling headphones",
#     "generated_output": "Silence everything. Hear only what matters.",
#     "reference_output": "Introducing our latest noise-cancelling headphones – experience pure sound, zero distractions."
#   },
#   {
#     "prompt": "Promote our smart LED light strip with app control",
#     "generated_output": "Control the vibe from your phone. Lights that follow your mood.",
#     "reference_output": "Transform your space with app-controlled LED lights. Your mood, your colours."
#   }
# ]
# Save as: eval_input.json

# -------------------------
# Load Data from JSON
# -------------------------
with open("eval_input.json", "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)

# -------------------------
# Prompt Template
# -------------------------
def create_eval_prompt(data_point):
    return f"""
You are a professional social media marketer. Evaluate the generated Instagram caption for the given product prompt.

Prompt: {data_point['prompt']}

Reference caption (ideal): {data_point['reference_output']}
Generated caption (model): {data_point['generated_output']}

Evaluate the generated caption based on the following:
1. Clarity – Is it easy to understand at a glance?
2. Creativity – Is it original, catchy, and scroll-stopping?
3. Alignment with Prompt – Does it reflect the intended product or theme?
4. Brand Tone – Does it match a modern brand voice (e.g. minimal, premium, playful)?
5. Call-to-Action (CTA) – If relevant, does it motivate the viewer to take action?

Respond in this format:
Feedback: <brief reasoning>
Clarity: <1-5>
Creativity: <1-5>
Alignment: <1-5>
Tone: <1-5>
CTA: <1-5 or N/A if not applicable>
""".strip()

# -------------------------
# Call gpt-4o to Evaluate
# -------------------------
def evaluate_caption(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# -------------------------
# Run Evaluation Loop
# -------------------------
results = []
for item in evaluation_data:
    eval_prompt = create_eval_prompt(item)
    try:
        feedback = evaluate_caption(eval_prompt)
        results.append({
            "Prompt": item["prompt"],
            "Generated Output": item["generated_output"],
            "LLM Feedback": feedback
        })
        time.sleep(1)  # Optional: avoid hitting rate limits
    except Exception as e:
        results.append({
            "Prompt": item["prompt"],
            "Generated Output": item["generated_output"],
            "LLM Feedback": f"Error: {str(e)}"
        })

# -------------------------
# 💾 Save or Display Results
# -------------------------
df = pd.DataFrame(results)
df.to_csv("llm_gpt4o_evaluation_results.csv", index=False)
df.head()
```

### 📄 `llm_gpt4o_evaluation_results.csv`

| Prompt | Generated Output | LLM Feedback |
| --- | --- | --- |
| Create a video script about Morpheus 1, an AI device claiming to control dreams using ultrasonic holograms, based on the style of the provided transcripts. | Okay, you're not going to believe this. There is an AI device that lets you control your dreams. This is an up-and-coming device called Morpheus 1, and this is how it is supposed to work. What Morpheus does is it takes the brain state as the input prompt, and based on the current state, it tries to alter the brain state, letting you into states of no symptoms. And the way it does it is by generating ultrasonic holograms. I'm honestly not sure about how good this product is going to be or whether it will actually do what it claims to do, but I'm surely excited by the possibilities that it offers.  | Feedback: A concise and impactful caption that matches the product and brand tone well. 

Clarity: 5
Creativity: 4
Alignment: 5
Tone: 5
CTA: N/A |
| Write a video script about Google's TextFX tool for assisting rappers and songwriters, following the style of the example transcripts | If you're a rapper or a songwriter, you're going to love this tool. It's called TextFX and it's by Google. And it's free. This tool helps accelerate your songwriting process in many different ways. I'll tell you how. Let's say that you're writing a verse in a rap and you're stuck at a point, which is a very common thing to happen. You can use TextFX in order to generate ideas in terms of the next word. You can find a metaphor, you can find a simile, you can find a rhyming word, and you can find words that sound similar to another word. It's quite insane. And when I saw the demo, it blew my mind. It was developed in collaboration with a rapper named Luke Fiasco. And in his own words, he says that this tool keeps rappers in the process. And being a hobbyist songwriter, I agree to a large extent. This is quite insane. For More AI stuff Follow 100X. | Feedback: Fun and catchy tone, though could mention it’s features more explicitly for clarity. 

Clarity: 4 
Creativity: 5 
Alignment: 4 
Tone: 5 
CTA: 5 |


# Other Eval framework 
- Deep Eval
  Open Source LLM Evaluation framework
  [deepeval/platform](https://deepeval.com/)

- Open ai Eval
  Open ai open source project
  [Openai eval/git](https://github.com/openai/evals)

- argilla-io - Distilable
  [argilla-io](https://github.com/argilla-io/distilabel)
