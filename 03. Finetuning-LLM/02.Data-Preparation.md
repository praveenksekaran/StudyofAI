# Data Preparation for LLMs

## Types of Datasets

**1️⃣ Instruction-Following (SFT) Datasets**

- **What:** Pairs of instructions + expected outputs.
- **Why:** Teach models how to follow commands, great for enterprise workflows (e.g., “Summarize this call” or “Extract action items.”).
- **Example:**
    
    ```json
    // Example 1
    {
      "instruction": "Summarize the meeting transcript.",
      "input": "CEO: Sales dropped 5%. CFO: Cut marketing by 10%...",
      "output": "Sales down 5%; plan to reduce marketing by 10%."
    }
    
    // Example 2
    {
      "instruction": "Summarize this meeting transcript.",
      "input": [
        {"speaker": "CEO", "text": "Good morning, everyone. The Q2 budget went over by 10%."},
        {"speaker": "CFO", "text": "That was due to unexpected travel expenses."},
        {"speaker": "VP Sales", "text": "We need a plan to reduce costs before Q3."},
        {"speaker": "CEO", "text": "Agreed. Let's brainstorm cost-saving ideas."}
      ],
      "output": "Q2 budget exceeded by 10% due to travel costs. Team agreed to find ways to cut expenses before Q3."
    }
    
    ```
    

---

**2️⃣ Chat/Dialogue Datasets**

- **What:** Multi-turn conversations
- **Why:** Fine-tune LLMs to maintain context across multiple turns.
- **Example Use Case:** An enterprise chatbot handling HR queries.

```json
[
  {"role": "system", "content": "You are a helpful meeting assistant."},
  {"role": "user", "content": "Summarize the Q2 budget discussion."},
  {"role": "assistant", "content": "Q2 budget overrun by 8%; discussion focused on cutting travel expenses."},
  {"role": "user", "content": "What were the proposed next steps?"},
  {"role": "assistant", "content": "Finance team to present revised budget next Monday."}
]
```

This **ChatML-style format** is the de-facto standard in many instruction/chat datasets (e.g., OpenAI, Anthropic, LLaMA chat examples).

---

**3️⃣ Domain-Specific Text Datasets**

- **What:** Collections of text related to a particular field (e.g., legal, medical, financial).
- **Why:** Adapt LLMs to specialized vocabularies.
- **Example:** Feeding your own company’s knowledge base to the model.
    
    Company SOPs, jargon-heavy meeting notes like:
    
    > “MRR, CAC, and ACV metrics discussed; ARR projections shared.”
    > 

```json
{
  "instruction": "Summarize this case review meeting.",
  "input": "Dr. Smith: Patient's HbA1c rose to 8.2%. Nurse: Increase Metformin dosage. Dr. Lee: Schedule follow-up in 4 weeks.",
  "output": "Patient HbA1c increased; plan to raise Metformin dosage and reassess in 4 weeks."
}
```

---

**4️⃣ Reasoning/Chain-of-Thought Datasets**

- **What:** Tasks requiring the model to show intermediate reasoning.
- **Why:** Improves logical accuracy for tasks like diagnostics or decision support.
- **Example:**
    
    ```
    Q: Should we hire more SDRs if leads dropped by 50%?
    A: Leads are down, so hiring more SDRs would not be effective until lead flow improves. Therefore, pause hiring.
    ```
    

## **Example: Fireflies-like Meeting Assistant**

We’ll build a meeting assistant that:

✅ Summarizes calls

✅ Generates action items

✅ Answers follow-up questions

| Dataset Type | What It Enables in Your Assistant |
| --- | --- |
| Instruction | Summarization, action item extraction, Q&A. |
| Chat/Dialogue | Context-aware responses, clarifications during multi-turn chats. |
| Domain-specific | Correct use of internal or industry-specific jargon. |
| Reasoning | Logical recommendations (“Should we delay launch if QA fails?”). |

## Dataset Formats & Popular Datasets

🔹 **Instruction-Input-Output (Alpaca/Vicuna-style)**

```json
{
  "instruction": "Summarize this meeting transcript.",
  "input": "Manager: The project is delayed. Developer: We need more QA resources...",
  "output": "Project delayed; more QA resources needed."
}
```

Most popular for single-turn tasks like summarization, Q&A, classification.

---

🔹 **ChatML / System-User-Assistant Format**

```json
[
  {"role": "system", "content": "You are a helpful meeting assistant."},
  {"role": "user", "content": "Summarize the Q2 discussion."},
  {"role": "assistant", "content": "Q2 revenue missed target; team plans to adjust marketing budget."}
]
```

Best for multi-turn conversations with context — used by OpenAI, ChatGPT, LLaMA2-Chat, Axolotl.

---

🔹 **OpenChat/ShareGPT Conversations Format**

```json
{
  "conversations": [
    {"from": "human", "value": "Summarize the meeting."},
    {"from": "gpt", "value": "The budget overran; team discussed cost cuts."},
    {"from": "human", "value": "What was the cause?"},
    {"from": "gpt", "value": "Unexpected travel expenses in Q2."}
  ]
}
```

Widely adopted in open-source chat datasets like ShareGPT, OpenChat, Vicuna conversations.

[**Examples of Open SFT Datasets**](https://github.com/mlabonne/llm-datasets?tab=readme-ov-file#-open-sft-datasets)

## Challenges in creating data

- Collecting real-world data can be time-consuming and expensive.
- Ensuring data quality, diversity, and lack of bias is difficult.
- Some domains have limited available data due to privacy or scarcity issues.
- Labeling data accurately often requires domain expertise.

### 1. Noisy or Low-Quality Data

- **Problem:** Transcripts often contain filler words, mishears, or irrelevant chit-chat:
    
    ```
    "Umm... okay, so like, yeah, the budget thingy... wait, what was I saying?"
    ```
    
- **Impact:** Models trained on this will generate incoherent or unprofessional outputs.
- **Solution Preview:** Preprocessing steps - cleaning fillers, punctuation, and speaker tags.

---

### 2. Duplicates & Redundant Examples

- **Problem:** Repeated conversations or same meeting transcribed multiple times.
- **Example:**
    - Same call saved by two team members → two identical transcripts → model overfits or memorizes.
- **Impact:** Skews model towards frequent examples → biases outputs.
- **Solution Preview:** Deduplication strategies with fuzzy matching or embeddings.

---

### 3. Data Decontamination

- **Problem:** Training data may contain company secrets, names, emails, phone numbers.
    
    ```
    "Hey John Doe, send that file to jane@company.com."
    ```
    
- **Impact:** Privacy violations, data leaks, regulatory/legal issues.
- **Solution Preview:** Automated PII redaction tools; regex + NER-based scrubbing.

---

### 4. Hallucinated or Incorrect Data

- **Problem:** Human-curated or auto-generated datasets can contain factual errors:
    - Incorrect metrics (“revenue grew 50%” when actual is 5%).
- **Impact:** Teaches model to generate unreliable summaries or reports.
- **Solution Preview:** Data validation pipelines, consistency checks, synthetic data from templates.

---

### 5. Imbalanced or Insufficient Data Diversity

- **Problem:** Most data might come from sales meetings → model performs poorly on engineering or HR calls.
- **Impact:** Model struggles outside dominant domain.
- **Solution Preview:** Augment data with synthetic or underrepresented meeting types.

---

### 6. Data Formatting & Consistency Issues

- **Problem:** Different meeting transcripts use inconsistent structures:
    - Some with “Speaker: text”, some without speaker tags.
    - Variations in timestamp placement, encoding (UTF-8/16).
- **Impact:** Breaks parsing scripts, causes fine-tuning errors.
- **Solution Preview:** Preprocessing pipelines to normalize formats.

---

### 7. Alignment with Evaluation Objectives

- **Problem:** Without clear evaluation goals, your dataset may not cover:
    - Summaries of different lengths
    - Different meeting styles (brainstorm vs. planning)
- **Impact:** Gaps in test coverage → unpredictable real-world performance.
- **Solution Preview:** Design dataset split and validation samples reflecting your assistant’s expected use cases.

## What is a good dataset?

![image.png](7cf050c9-ea9f-4376-bb16-708a1d157a6e.png)

### 1. Accuracy

- Data outputs should be factually correct and free of typos.
- **Example:**
    - ❌ Bad: “Q2 revenue increased 50%” (when it was 5%).
    - ✅ Good: “Q2 revenue increased 5%.”
- **Why it matters:** Inaccurate data teaches the model to generate wrong summaries → massive trust issues.

---

### 2. Diversity

- Dataset covers a wide range of topics, styles, and meeting types.
- **Example:**
    - Include sales calls, engineering stand-ups, HR discussions, executive QBRs.
    - Vary writing styles: concise bullet summaries, detailed paragraphs.
- **Why it matters:** Prevents the model from failing when exposed to unfamiliar meeting topics.

---

### 3. Complexity

- Examples include complex tasks that challenge the model to reason or explain.
- **Example:**
    - Chain-of-thought tasks:
        
        ```
        Q: Why did churn increase?
        A: Churn rose because product bugs delayed launch, causing customer dissatisfaction.
        ```
        
    - Multi-step summarization: Generate summary → extract action items → assign owners.
- **Why it matters:** Prepares the model to handle real-world questions beyond simple commands.

---

### 4. Consistency & Alignment

- Dataset follows a consistent format, style, and aligns with your model’s intended purpose.
- **Example:**
    - If your assistant is formal → dataset should use formal language throughout.
    - If your assistant is casual → dataset should match that tone.
- **Why it matters:** Prevents the model from switching styles unpredictably.

## Creating SFT Datasets: A Recipe

### 📝 Step 1: Collect Raw Data

- Gather text from open datasets, internal documents, APIs, transcripts, or scrape public sources (respecting licenses & privacy).
- Aim for **at least 500–1000 unique meeting transcripts** to start.
- Example sources: Transcripts of sales calls, marketing reviews, executive QBRs.

**Where do we get this data from?**

> 1. Export transcripts from your organization’s meeting software (Zoom, MS Teams, Google Meet, Fireflies itself if you have it). Save them as `.txt` or `.json` files.
2. Scrape or download publicly available meeting transcripts.
3. Generate synthetic meeting transcripts from SOTA LLM (GPT/Claude/Gemini).
4. Transcribe audio recordings using OpenAI Whisper.
> 

### Example Raw Transcript

```
[00:00:01] CEO: Welcome, team. Let’s discuss Q2 numbers.
[00:00:15] CFO: Revenue fell 5% compared to Q1 due to increased churn.
[00:00:32] VP Sales: We lost two major accounts this quarter.
[00:00:50] CEO: What are our immediate next steps?
```

```json
// Multiple Meeting raw transcriptions
raw_meetings = [
    [
        "[00:00:05] CEO: Um, so the Q2 numbers are not great.",
        "[00:00:10] CFO: Revenue fell 5%.",
        "[00:00:15] CEO: We need to cut marketing spend."
    ],
    [
        "[00:00:03] HR: Employee turnover rose 7% last quarter.",
        "[00:00:08] CEO: Any ideas why?",
        "[00:00:12] HR: Feedback suggests lack of career growth."
    ],
    [
        "[00:00:01] CTO: Vendor delays are impacting release timeline.",
        "[00:00:05] Dev Lead: Estimated two-week slip on Feature X.",
        "[00:00:10] CEO: Keep me updated daily."
    ],
    [
        "[00:00:02] VP Sales: Churn increased by 3% last month.",
        "[00:00:06] CEO: Let’s analyze churn reasons before next week.",
        "[00:00:09] VP Sales: Agreed, team is already collecting data."
    ]
]

```

---

### 🧹 Step 2: Clean the Data

- Normalize text: fix encoding issues, standardize punctuation.
- Remove irrelevant sections: disclaimers, footers, disclaimers.
- De-identify or redact sensitive information.
- **Example**:
    - Remove fillers: “um”, “uh”, “you know.”
    - Standardize speaker tags: always use `Speaker: text` or structured JSON.
    - Redact PII: emails, phone numbers, sensitive names.

## Python Script:

```python
import re
import json

def clean_fillers(text):
    """
    Remove common fillers like 'um', 'uh', 'like', etc.
    """
    fillers = r"\b(um+|uh+|like|you know|okay|so)\b"
    return re.sub(fillers, "", text, flags=re.IGNORECASE)

def normalize_speaker_tags(line):
    """
    Remove timestamps like [00:00:15] and keep 'Speaker: text'.
    """
    # Remove leading timestamps
    line = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", line)
    return line.strip()

def redact_pii(text):
    """
    Replace emails and 10-digit phone numbers with placeholders.
    """
    text = re.sub(r"\S+@\S+", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\b\d{10}\b", "[REDACTED_PHONE]", text)
    return text

def clean_transcript(raw_lines):
    """
    Apply all cleaning functions to each line in a transcript.
    """
    cleaned_lines = []

    for line in raw_lines:
        line = normalize_speaker_tags(line)
        line = clean_fillers(line)
        line = redact_pii(line)
        line = re.sub(r"\s+", " ", line).strip()  # Normalize whitespace
        if line:  # Skip empty lines
            cleaned_lines.append(line)

    return cleaned_lines

if __name__ == "__main__":
    # 🔎 Example raw transcript lines (simulate reading from a file)
    raw_transcript = [
        "[00:00:05] CEO: Um, okay, so like, the Q2 numbers, uh, they are not great.",
        "[00:00:15] CFO: Yeah, um, revenue fell 5%. john.doe@company.com knows more.",
        "[00:00:20] CEO: You know, we need to address this asap. Call me at 9876543210."
    ]

    cleaned_transcript = clean_transcript(raw_transcript)

    # Print cleaned lines
    print("\n".join(cleaned_transcript))

    # 🔎 Save cleaned transcript as JSON array of lines
    with open("cleaned_transcript.json", "w") as f:
        json.dump(cleaned_transcript, f, indent=2)

```

**After cleaning:**

```json
[
  "CEO: The Q2 numbers are not great.",
  "CFO: Revenue fell 5%. [REDACTED_EMAIL] knows more.",
  "CEO: We need to address this asap. Call me at [REDACTED_PHONE]."
]
```

---

### 🔄 Step 3: Deduplicate & Decontaminate

- Detect and remove near-duplicates to prevent overfitting.
- Decontaminate by comparing against evaluation datasets (avoiding train-test leakage).
- Example: Identify and remove near-duplicate transcripts (using fuzzy matching or embeddings)

**Deduplication: Use embeddings + cosine similarity**

Detect duplicates even with slight variations by comparing semantic similarity  i.e., similarity in meaning, not exact words.

```python
from sentence_transformers import SentenceTransformer, util

# Sample cleaned transcripts (concatenate lines for simplicity)
transcripts = [
    "CEO: The Q2 numbers are not great. CFO: Revenue fell 5%.",
    "CEO: The Q2 numbers are bad. CFO: Revenue decreased by 5 percent.",
    "Manager: Project Alpha is on schedule. Engineer: We finished the first milestone."
]

# Load pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for each transcript
embeddings = model.encode(transcripts, convert_to_tensor=True)

# Compute pairwise cosine similarity matrix
cosine_scores = util.cos_sim(embeddings, embeddings)

# Threshold for duplicates (tune between 0.8–0.95 depending on your data)
threshold = 0.9

duplicates = set()
for i in range(len(transcripts)):
    for j in range(i + 1, len(transcripts)):
        if cosine_scores[i][j] > threshold:
            print(f"Duplicate found:\n- {transcripts[i]}\n- {transcripts[j]}\n")
            duplicates.add(j)

# Filter out duplicates
unique_transcripts = [t for idx, t in enumerate(transcripts) if idx not in duplicates]

print("Unique transcripts after deduplication:")
for t in unique_transcripts:
    print("-", t)
```

**Decontamination: Use Embeddings or Fuzzy Matching**

Compares sequences of characters or tokens. Great for catching small typographical differences or formatting changes. When you need to detect exact or near-exact duplicates with minor edits.

```python
from rapidfuzz import fuzz

# Example eval set sample
eval_samples = [
    "Summarize the meeting where revenue declined and marketing spend was discussed."
]

# Your cleaned dataset samples
dataset_samples = [
    "Summarize the meeting where revenue fell 5% and marketing was discussed.",
    "Summarize the meeting about hiring new engineers."
]

for eval_sample in eval_samples:
    for idx, dataset_sample in enumerate(dataset_samples):
        score = fuzz.token_set_ratio(eval_sample, dataset_sample)
        if score > 85:  # threshold depends on your tolerance
            print(f"Decontamination match:\n- Eval: {eval_sample}\n- Dataset: {dataset_sample}\n- Score: {score}")
```

### Outputs:

```
Transcript 1:
CEO: The Q2 numbers are not great. CFO: Revenue fell 5%.

Transcript 2:
CEO: The Q2 numbers are bad. CFO: Revenue decreased by 5 percent.

Cosine similarity score: 0.98  (scale: -1 to 1)
=> High semantic similarity → flagged as duplicates

Fuzzy token set ratio score: 82  (scale: 0 to 100)
=> Moderate lexical similarity → may or may not flag as duplicate depending on threshold
```

---

### 🏷️ Step 4: Define Tasks & Annotate Examples

- For each example, write a clear **instruction** describing what the model should do.
- Pair it with an **input** (text or question) and the **expected output** (answer, summary, classification label, etc.).
- For each example, create tasks:
    - Summarization
    - Action item extraction
    - Follow-up Q&A

1. Ensure instructions are clear and concise

```
**// Summarization**
"Summarize the meeting transcript."

// Action Items
"Extract action items from the meeting."
```

1. Pair instructions with cleaned input and expected output (automate where you can)

```python
import json

cleaned_transcripts = [
    [
        {"speaker": "CEO", "text": "The Q2 revenue fell 5%."},
        {"speaker": "CFO", "text": "Marketing overspend led to budget issues."}
    ]
]

samples = []
for transcript in cleaned_transcripts:
    sample = {
        "instruction": "Summarize the meeting transcript.",
        "input": transcript,
        "output": "[WRITE SUMMARY HERE]"  # Placeholder for manual/LLM annotation
    }
    samples.append(sample)

with open("annotated_dataset.json", "w") as f:
    json.dump(samples, f, indent=2)
```

1. Save everything in a consistent format (**Alpaca/Vicuna/ChatML/OpenChat**)

### Output

```json
{
  "instruction": "Extract action items from the meeting transcript.",
  "input": [
    {"speaker": "VP Engineering", "text": "Complete code review by Wednesday."},
    {"speaker": "QA Lead", "text": "Finish test cases before Friday release."}
  ],
  "output": "- Complete code review by Wednesday.\n- Finish test cases before Friday release."
}
```

---

### 🧠 Step 5: Add Diversity & Complexity

- Include varied topics, speaking styles, or scenarios reflecting real-world use cases.
- Add examples requiring multi-step reasoning (chain-of-thought) if relevant.
- Add edge cases where speaker contradict each other.

**Augment Data (Optional)**

- Create synthetic examples by rephrasing instructions or inputs.
- Use controlled generation with existing LLMs to expand coverage (but validate outputs!).
- Example: Vary meeting length, complexity, topics.

---

### 📊 Step 6: Evaluate Quality

- Manually inspect random samples: check factual accuracy, clarity, format consistency.
- Calculate basic dataset stats: length distribution, task type balance, label frequencies.
- Optional: involve domain experts for domain-specific data.

**In Practice:**

**Random Sampling: Select ~5–10% of samples randomly.**

Checklist:

✔ Instruction clear and concise?

✔ Input cleaned (no PII, consistent speaker tags)?

✔ Output correct, coherent, and complete?

**Automated Format Checks**

Write a script to scan every sample:

- Check for missing fields: `instruction`, `input`, `output`.
- Verify `input` is a list of dicts with `speaker` + `text`.
- Detect empty outputs.

Example:

```python
import json

with open("synthetic_meeting_dataset.json") as f:
    data = json.load(f)

for i, sample in enumerate(data):
    if not all(k in sample for k in ["instruction", "input", "output"]):
        print(f"Sample {i} missing fields!")
    if not isinstance(sample["input"], list) or not sample["input"]:
        print(f"Sample {i} has bad input format!")
    if not sample["output"]:
        print(f"Sample {i} has empty output!")
```
Also try llm_judge.ipynb, sometimes both
---

### 🔁 Step 7: Iterate

- Fine-tune a model → evaluate → identify failure modes → adjust or collect more data.
- Repeat until you reach target performance.

### Bonus:

## [**Distilabel**](https://github.com/argilla-io/distilabel)

✅ Annotates instructions & outputs

✅ Adds diversity & complexity

✅ Augments data

All with a **single script** you can drop into your workflow.

**Also try augmentoolkit **

---

### Script

```python
import json
from distilabel.tasks import GenerationTask
from distilabel.policies import GenerationPolicy
from distilabel.executors import LocalExecutor

# 1️⃣ Load your cleaned transcripts (e.g., output of cleaning/deduplication)
# Expected format: [{"input": "CEO: Revenue fell 5%. CFO: Cut marketing by 10%."}, ...]
with open("cleaned_transcripts.json") as f:
    cleaned_data = json.load(f)

# 2️⃣ Define multiple tasks for annotation + diversity
summarize_task = GenerationTask(
    name="summarize",
    prompt_template=(
        "You are a helpful meeting assistant. Summarize the following meeting transcript:\n\n{input}"
    )
)

action_items_task = GenerationTask(
    name="generate_action_items",
    prompt_template=(
        "You are a meeting assistant. Extract action items from the following transcript:\n\n{input}"
    )
)

qa_task = GenerationTask(
    name="answer_followup",
    prompt_template=(
        "You are a meeting assistant. Based on the following transcript, what should the team focus on next?\n\n{input}"
    )
)

# 3️⃣ Create a generation policy combining these tasks
policy = GenerationPolicy(
    tasks=[summarize_task, action_items_task, qa_task],
    model="gpt-4o",                 # Replace with your OpenAI-compatible endpoint
    max_tokens=512,
    temperature=0.7,                # Add diversity with mild randomness
)

# 4️⃣ Create the executor to run the policy on your dataset
executor = LocalExecutor(policy=policy)

# 5️⃣ Execute: this generates multiple augmented examples per input
results = executor.run(cleaned_data)

# 6️⃣ Save the augmented dataset in consistent instruction-input-output format
augmented_dataset = []
for res in results:
    input_text = res['input']
    for task_name, task_result in res['results'].items():
        augmented_dataset.append({
            "instruction": f"{task_name.replace('_', ' ').capitalize()} the meeting transcript.",
            "input": input_text,
            "output": task_result['output']
        })

with open("distilabel_augmented_dataset.json", "w") as f:
    json.dump(augmented_dataset, f, indent=2)

print(f"Generated {len(augmented_dataset)} augmented examples saved to 'distilabel_augmented_dataset.json'.")

```

---

### What does this script do?

- **Reads your cleaned transcripts** — one transcript per example.
- For each transcript, runs **three tasks** (`summarize`, `generate_action_items`, `answer_followup`) → multiplies your dataset size by 3.
- Outputs consistent JSON objects with:

```json
{
  "instruction": "Summarize the meeting transcript.",
  "input": "CEO: Revenue fell 5%. CFO: Cut marketing by 10%.",
  "output": "Revenue dropped 5%; plan to reduce marketing costs by 10%."
}
```

- Saves everything in a single dataset file ready for SFT.
