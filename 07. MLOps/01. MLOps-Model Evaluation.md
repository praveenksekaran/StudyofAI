# Introduction

MLOps for Gen AI can be defined as a set of practices, techniques and technologies that extend MLOps principles to the operationaliization of generative AI applications.

<img width="836" height="408" alt="image" src="https://github.com/user-attachments/assets/059907ff-e49f-4787-9d38-ed5e7a6f2aee" />

<img width="807" height="418" alt="image" src="https://github.com/user-attachments/assets/a04000f3-4d4c-4306-90c0-48d7f60f353e" />

<img width="877" height="415" alt="image" src="https://github.com/user-attachments/assets/ac87617d-ca7f-4ee7-a793-ee13abad79d6" />

# Challenges

### 1. Core Concept
Existing MLOps investments are not obsolete; they are the foundation for Generative AI. The goal is to integrate LLMs (Large Language Models) into the MLOps ecosystem while addressing unique infrastructure, tuning, and evaluation challenges.

### 2. Challenge 1: Infrastructure & Discovery
Generative models require massive computational power (GPUs/TPUs) for training and deployment.

#### Vertex AI Solutions:
*   **Model Garden:** A single gateway to discover Google, open-source, and third-party models. It allows for immediate experimentation (classification, summarization, etc.) using simple text instructions.
*   **Generative AI Studio:** A fully managed environment for testing and designing prompts without worrying about underlying infrastructure.
*   **Vertex Training & Prediction:** Scalable compute infrastructure where Google manages the backend, allowing teams to focus on model logic rather than hardware.

### 3. Challenge 2: Customization & Tuning
Generic models often need "nudges" to perform specific tasks or align with domain-specific knowledge.

#### Tuning Methods:
*   **Supervised Tuning:** Used when you have a dataset with well-defined, correct outputs.
*   **Reinforcement Learning with Human Feedback (RLHF):** Ideal for subjective tasks like chat or summarization where "correctness" is harder to define numerically.
*   **Data Curation:** Enhancing pre-trained models by adding domain-specific data to the mix.

### 4. Challenge 3: Management of New Artifacts
GenAI introduces artifacts that traditional ML doesn't typically handle, such as prompts and vector embeddings.

#### Management Tools:
*   **Prompt Management:** Tools like **LangChain** and **Weights & Biases** assist in designing, analyzing, and debugging complex instructions.
*   **Embedding Management:** **Vertex AI Feature Store** manages dense vector representations used for similarity matching and search.
*   **Adaptive Layer Management:** **Vertex Model Registry** tracks "adapter layers" (small weight updates) separately from the massive foundation models.
*   **Tuning Job Management:** **Vertex AI Pipelines** ensures lineage and reproducibility for the fine-tuning process.

### 5. Challenge 4: Evaluation & Monitoring
Unlike traditional ML (where you might look at simple accuracy), GenAI requires evaluating unstructured text and images.

#### Strategies:
*   **Metric Evolution:** Moving toward metrics for **fluency**, **factuality**, and **brand reputation**.
*   **Vertex AI Evaluation Services:** Uses an evaluation dataset (prompt + ground truth) to compute performance.
*   **Safety Monitoring:** Automated safety scores across 10+ categories to flag inappropriate content.
*   **Recitation Checking:** A built-in scanner that checks outputs against the web or code repositories to prevent the use of unoriginal content (plagiarism prevention).



### 6. Challenge 5: Enterprise Data Integration
Models need to "know" your company's private data to be useful and to reduce "hallucinations."

#### Integration Capabilities:
*   **Vector Search:** Facilitates semantic knowledge comparison across enterprise text and images.
*   **Grounding:** Forces the model to generate responses based *only* on provided enterprise data.
*   **Vertex Extensions:** Connects models to real-time data and allows them to perform real-world actions (e.g., booking a flight or querying a live database).



# Model Evaluation in Machine Learning Operations

[Model evaluation in Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/evaluation/introduction)

### 1. What is Model Evaluation?
Model evaluation is the process of assessing a machine learning model's performance. It serves as a quality check to ensure the model:
* **Is Accurate:** Does it make correct predictions?
* **Is Reliable:** Does it perform consistently on new, unseen data?
* **Is Aligned:** Does it effectively support the intended business goals?

### 2. Why Model Evaluation Matters
Evaluation is pivotal for five main reasons:
* **Performance Measurement:** Provides clear metrics to understand how well the model works.
* **Generalization:** Ensures the model functions in real-world scenarios, not just on training data.
* **Model Selection:** Helps identify the most effective model from multiple candidates, optimizing resources.
* **Continuous Improvement:** Tracks performance after deployment to identify degradation or data shifts, signaling when retraining is necessary.
* **Informed Decision Making:** Gives stakeholders the confidence to approve deployments or updates.

### 3. When Evaluation Occurs
Evaluation is a continuous cycle focused on two primary stages:
1.  **After Training:** Metrics are examined to determine if the model is ready for deployment.
2.  **After Deployment:** Known as **Continuous Evaluation**, this monitors the model against new data to catch performance declines.

### 4. How Evaluation is Done: Techniques vs. Metrics
Evaluating a model involves two distinct concepts:

#### A. Evaluation Techniques (The "Process")
The overarching procedures used to test the model (e.g., how the data is split).
* **Holdout Validation**
* **K-fold Cross-Validation**
* **Leave-one-out Cross-Validation**

#### B. Evaluation Metrics (The "Score")
The numerical scoring system used to quantify specific tasks.
* **Classification:** Accuracy, Precision, Recall, F1-score.
* **Regression:** Mean Squared Error (MSE), R-squared.
* **NLP:** BLEU, ROUGE.

> **The Cake Analogy:** > * **Techniques** are the process of baking (mixing, temperature, timing).
> * **Metrics** are how you judge the result (taste, texture, appearance).

### 5. Factors for Selecting Evaluation Methods
Selection is not "one size fits all." Consider:
* **Model Type:** (Classification, Regression, Ranking).
* **Project Goal:** Which performance aspects are most critical?
* **Dataset Size:** Small datasets favor simple techniques; large datasets can handle complex methods.
* **Computational Cost:** Balancing accuracy with the resources required.
* **Bias-Variance Trade-off:** Using techniques like bootstrapping to find over-fitting or under-fitting.
* **Cost of Errors:** How critical is a mistake? (e.g., medical diagnosis vs. product recommendation).
* **Data Balance:** Ensuring metrics account for imbalanced classes.

### 6. Stakeholders: Who Cares About the Results?
* **Data Scientists/ML Engineers:** To fine-tune and choose the best models.
* **Business Leaders:** To understand the impact on revenue and customer satisfaction.
* **Software Developers:** To ensure smooth system integration and reliability.
* **End Users:** To ensure the tool they interact with is trustworthy.
* **Regulatory Bodies:** To ensure adherence to ethical AI standards.
* **Researchers:** To improve methods and develop field best practices.

## MLOps and Vertex AI: Operationalizing Machine Learning

### 1. Defining MLOps
Machine Learning Operations (MLOps) is the practice of combining **DevOps** principles with **Machine Learning**. It bridges the gap between model development and production-ready operationalization.

#### Core Pillars:
*   **Collaboration:** Uniting data scientists, engineers, and business stakeholders.
*   **Iteration:** Continuously refining models via regular evaluation and data updates.
*   **Systemization:** Providing a structured framework for reliability, reproducibility, and model management.
*   **Goal:** To create accurate, robust, and business-driven ML models.

### 2. The Role of Vertex AI
Vertex AI acts as an end-to-end platform that unifies Google Cloud’s AI services to streamline the development lifecycle.

*   **Unified Platform:** Simplifies building, deploying, and scaling models.
*   **Lifecycle Management:** Streamlines the transition from experimentation to production.
*   **Governance:** Achieves mature MLOps through comprehensive evaluation capabilities.

### 3. Integrating Evaluation into the Lifecycle
Intentional evaluation must be embedded at every stage (building, training, and deployment) to mitigate risks.

#### Benefits of Embedded Evaluation:
*   **Risk Mitigation:** Early identification of malfunctions and performance bottlenecks.
*   **Bias Detection:** Ensuring fairness and ethical AI standards.
*   **Reliability:** Maintaining a high-quality user experience and consistent performance.

### 4. Enhancing MLOps Maturity
Increasing "MLOps Maturity" involves moving from manual processes to automated, integrated workflows.

#### How Vertex AI Supports Maturity:
*   **Automated Workflows:** Integrates training, validation, and deployment phases into a seamless pipeline.
*   **Evaluation at Scale:** Runs iterative evaluations on new datasets across large-scale environments.
*   **Advanced Visualizations:** Provides tools to compare model versions and select the optimal candidate for production.
*   **Granular Analysis:** Assesses performance across different "data slices" and annotations for a deeper understanding of model behavior.

### 5. Key Outcomes of Using Vertex AI
*   **Robust Feedback Loops:** Establishes a continuous cycle of improvement between deployment and retraining.
*   **Fairness Checks:** Ensures models are reliable and effective across diverse populations.
*   **Accelerated Journey:** Increases model quality while speeding up the transition to a production-ready environment.

## Model Evaluation Challenges and Solutions
Here are the precise notes from the transcript regarding **Model Evaluation Challenges and Vertex AI Solutions**, formatted in Markdown.

### 1. Common Challenges in Model Evaluation
Even high-performing models face obstacles that can impact their trustworthiness and reliability.

#### Data-Related Issues
*   **Overfitting:** The model excels on training data but fails to generalize to new, unseen data.
*   **Data/Concept Drift:** Real-world data distributions change over time, leading to performance decline.
*   **Lack of Representative Data:** Training data that fails to cover the full spectrum of real-world scenarios results in inaccurate applications.

#### Metric & Interpretability Issues
*   **Misleading Metrics:** Relying solely on one metric (like accuracy) can hide poor performance in minority classes.
*   **Goal Mismatch:** Choosing metrics that do not align with the specific project objectives.
*   **Black Box Models:** Complex models (e.g., Deep Neural Networks) are difficult to decipher, making it hard to explain specific predictions.

#### Ethical & Operational Issues
*   **Bias and Fairness:** Inherent biases in data can be amplified by the model, leading to discriminatory outcomes.
*   **Integration Hurdles:** Technical and communication gaps when moving a model into a larger production system.

### 2. Mitigation Strategies
To ensure reliable performance, practitioners should employ these specific techniques:

| Challenge | Mitigation Strategy |
| :--- | :--- |
| **Performance Assessment** | Use multi-dimensional metrics: Precision, Recall, F1-score, and AUC-ROC. |
| **Reliability** | Use Meticulous validation and splitting (Stratified sampling, Cross-validation). |
| **Overfitting** | Apply **Regularization**, **Dropout** (redundant representations), or **Early Stopping**. |
| **Bias/Fairness** | Implement fairness metrics to identify and neutralize discriminatory patterns. |
| **Explainability** | Use tools like **LIME**, **SHAP**, or feature importance to interpret "black box" models. |

### 3. Vertex AI Evaluation Capabilities
Vertex AI streamlines the complex process of evaluating, comparing, and selecting models at scale.

#### Core Requirements for Vertex AI Evaluation:
1.  **Trained Model:** Created via AutoML or custom training.
2.  **Batch Prediction Output:** Results obtained from running a batch job on the trained model.
3.  **Ground Truth Dataset:** Correctly labeled data (typically the test dataset) for comparison.

#### Key Features:
*   **Advanced Visualization:** Compare different model versions and evaluate performance across specific data "slices."
*   **Continuous Evaluation:** Automatically monitor deployed models with incoming data to signal when retraining is needed.
*   **Broad Support:** Handles **Classification, Regression, and Forecasting** across **Image, Text, Video, and Tabular** data.

### 4. The Vertex AI Evaluation Workflow
1.  **Train:** Use AutoML or custom code.
2.  **Predict:** Execute a batch prediction job.
3.  **Prepare Ground Truth:** Organize correctly labeled data.
4.  **Evaluate:** Initiate the evaluation job to compare predictions vs. ground truth.
5.  **Analyze:** Review metrics to understand strengths and weaknesses.
6.  **Iterate:** Refine the model based on insights and re-run evaluations to optimize.

## Challenges of evaluating the generative AI tasks - Intro

### 1. Predictive AI vs. Generative AI
While both share MLOps principles, their fundamental goals and evaluation needs differ:
*   **Predictive (Non-Generative) AI:** Focuses on analyzing existing data to make specific predictions, classifications, or decisions (closed-ended).
*   **Generative AI:** Learns patterns from vast datasets to create entirely new, open-ended content (text, images, music, code).

### 2. Defining LLMs within Generative AI
It is important to distinguish between the broad field and the specific subset:
*   **Generative AI:** An umbrella term for any AI that creates content across multiple modalities (e.g., images, music, video).
*   **Large Language Models (LLMs):** A specific subset of Generative AI specialized in language-based tasks such as translation, summarization, and text generation.

### 3. The "LLM Block": Key Components of the Lifecycle
Evaluating an LLM is a paradigm shift; it moves from optimizing simple parameters to orchestrating an intricate "block" of interacting components:

<img width="837" height="418" alt="image" src="https://github.com/user-attachments/assets/e6890655-9314-4bc4-8226-5257c2fcc41f" />


#### The Core Engine
*   **The LLM:** The central reasoning engine, typically accessed via APIs (e.g., Google) or open-source alternatives (e.g., Mistral).

#### Data and Context
*   **Data Sources:** Relational, graph, or vector databases that provide context, especially for **Retrieval-Augmented Generation (RAG)**.
*   **Memory:** A dynamic source that stores past interactions to maintain context for subsequent user requests.

#### Instruction and Interaction
*   **Prompt Templates:** Standardized, version-controlled instructions managed like code (e.g., prompt files) to guide model behavior.
*   **Tools:** External integrations that allow the model to make API calls, execute code, or interact with other systems.

#### Control and Safety
*   **Agent Control Flow:** Logic that allows the model to iteratively refine its approach to a task until a stopping criterion is met.
*   **Guardrails:** Safety mechanisms that filter output (via keyword detection or secondary models) before it reaches the user, sometimes triggering human review.

### 4. The Evaluation Paradigm Shift
Evaluation in the LLM lifecycle is more complex than traditional machine learning:
*   **Traditional ML Evaluation:** Focuses primarily on optimizing parameters and hyperparameters to improve predictive performance on unseen data.
*   **LLM Evaluation:** Focuses on orchestrating the interaction between the core engine and its various components (prompts, tools, and data) to ensure **reliability and quality** in a creative design space.

## Challenges in Evaluating Large Language Models (LLMs)

### 1. Data-Related Challenges
Unlike traditional machine learning, which begins with substantial datasets, LLM evaluation often faces data scarcity and quality issues.

*   **Lack of Initial Data:** Generative models can start with zero data. While this speeds up development, it makes it difficult to establish a "ground truth" or clear benchmark for success.
*   **Data Contamination:** Because foundation models are trained on massive, diverse data sources, test data may accidentally be included in the training set. This undermines the validity of benchmarks.
*   **Limited Reference Data:** Traditional metrics (like BLEU or ROUGE) require "perfect" reference answers. However, in creative or open-ended tasks, there are often multiple correct answers, making high-quality reference data hard to find.
*   **Quality Criteria:** Defining what actually constitutes a "good" dataset for LLM evaluation remains an open research question.

### 2. Model Complexity & Decision Space
The scale and architecture of LLMs create a massive "decision space" that complicates evaluation.

*   **Configuration Choices:** Practitioners must navigate a complex range of choices across training, model selection, customization, and in-context learning.
*   **Interpretability:** The internal workings of these models are difficult to decipher, making it hard to explain *why* a specific configuration led to a specific output.

### 3. Bias, Fairness, and Ethics
LLMs risk amplifying social inequalities if not carefully monitored.

*   **Social Inequality:** Models can produce unfair outcomes based on biases inherited from their training data.
*   **Mitigation:** Evaluation must include specific techniques for **bias detection and mitigation** to ensure ethical deployment.

### 4. Generalization vs. Real-World Use
There is often a gap between controlled test environments and actual deployment.

*   **Standardized Tests vs. Reality:** Benchmarks provide rankings, but the "messy" nature of the real world is far more unpredictable than a controlled benchmark.
*   **Contextual Complexity:** Success in a benchmark does not always guarantee success in a complex, real-world scenario.

### 5. Security and Robustness
LLMs introduce new attack vectors that traditional predictive models do not face.

*   **Adversarial Attacks:** Malicious actors can craft specific inputs to force the model into generating harmful or incorrect outputs.
*   **Data Poisoning:** The vulnerability to manipulated predictions or corrupted training data highlights a significant gap in current evaluation methodologies.

### 6. Subjectivity and Rapid Evolution
Evaluation is moving away from purely technical, objective scores.

*   **Creative Subjectivity:** Evaluating creative outputs (stories, poetry, chat) is inherently subjective and requires more nuanced methodologies than single-answer problems.
*   **Adapting to Change:** New evaluation methods are emerging rapidly, requiring organizations to stay adaptable and constantly update their assessment practices.

This summary covers the spectrum of evaluation types and specific metrics used to assess Large Language Models (LLMs), moving from simple binary checks to complex diversity and user-centric measurements.

## Beyond Accuracy: Mastering Evaluation Metrics for Generative AI

### 1. Spectrum of Evaluation Types
Evaluation methods range in complexity, each providing different levels of insight into model performance.

*   **Binary Evaluation:** A simple pass/fail or yes/no judgment (e.g., spam detection, content moderation). Easy to implement but lacks nuance.
*   **Categorical Evaluation:** Provides multiple options (e.g., Positive/Neutral/Negative or 1–5 star ratings). Offers more detail but is harder to define clearly.
*   **Ranking Evaluation:** Compares multiple model outputs to determine relative quality based on preference. Excellent for identifying top performers but is resource-intensive.
*   **Numerical Evaluation:** Provides objective, quantitative scores (e.g., accuracy %, BLEU, ROUGE). Highly comparable but may miss qualitative nuances.
*   **Text Evaluation:** Uses human-generated comments and critiques. Offers rich, domain-expert insights but is difficult to scale.
*   **Multitask Evaluation:** Combines various judgment types (quantitative and qualitative) for a comprehensive view of model capabilities across different tasks.

### 2. Key Performance Metrics
To properly evaluate LLMs, practitioners use a combination of traditional NLP metrics and modern, human-centric assessments.

#### Lexical Similarity (Vocabulary Check)
*Measures how closely the output matches a human reference text.*
*   **BLEU:** Focuses on precision (correct words).
*   **ROUGE:** Focuses on recall (capturing all necessary information).
*   **METEOR:** Balances both precision and recall.

#### Linguistic Quality (Structure & Clarity)
*Checks for grammar, fluency, and coherence.*
*   **BLEURT:** A BERT-based metric for text generation.
*   **Perplexity:** Measures how well a model predicts the next word. **Note:** Lower perplexity usually means better fluency, but it does not measure safety or relevance.

#### Specialized Evaluations
*   **Task-Specific:** Does the model address the assignment? (e.g., "Exact Match" for Q&A).
*   **Safety & Fairness:** Detecting bias, hate speech, or offensive content via human review or specialized tools.
*   **Groundedness:** Fact-checking to ensure the model isn't "hallucinating" or making things up.
*   **User-Centric:** Measuring user satisfaction, engagement, and helpfulness through surveys and completion rates.

### 3. Diversity Metrics
Diversity is crucial to ensure models do not produce repetitive or formulaic "generic" responses.

*   **Distinct-n:** Calculates the number of unique sequences of words (n-grams).
*   **Entropy:** Measures the unpredictability of output; higher entropy generally indicates higher creativity.
*   **Self-BLEU:** Measures the model's output against *itself*. A lower score suggests more diverse (less repetitive) responses.
*   **MAUVE:** Compares the word distribution of AI text against a large collection of human writing.
*   **Coverage:** Tracks how many concepts from a reference dataset are included in the model's output.

### 4. Key Takeaways for Practitioners
*   **The "Essay" Analogy:** Think of evaluation like a teacher grading an essay—you must check vocabulary (lexical), grammar (linguistic), adherence to the prompt (task-specific), and factual accuracy (groundedness).
*   **Balance is Critical:** High diversity (creativity) can sometimes come at the cost of coherence or relevance.
*   **Holistic Approach:** No single metric is sufficient. Automated metrics should be supplemented with human judgment to understand the full richness of generated text.

This summary details the best practices for evaluating Large Language Models (LLMs) throughout their lifecycle, using a trend-identification project to illustrate real-world challenges.

## Best Practices for LLM Evaluation

### 1. The Two Phases of Evaluation
Evaluation is not a one-time event; it is an ongoing process categorized into two distinct stages:

*   **Pre-production:** Focuses on foundational design choices.
    *   Designing and testing prompt templates.
    *   Selecting the appropriate base model.
    *   Optimizing customization (tuning parameters).
*   **In-production:** Focuses on real-world reliability.
    *   Continuous performance monitoring.
    *   Ensuring ongoing alignment with cultural, operational, and business objectives.

### 2. Case Study: "Trend Identification" Project
To illustrate evaluation needs, consider a media company automating news trend spotting:
*   **The Task:** Sifting through news, social media, and comments to group articles, analyze sentiment, and extract keywords.
*   **Evaluation Factors:** Managing at least two different models processing 1,000+ articles daily using multiple metrics.
*   **Key Hurdles identified:**
    *   **Scalability:** Building a framework that handles complex outputs at high volume.
    *   **Validation:** Accurately calculating metrics across vast datasets.
    *   **Reusability:** Creating a framework that adapts to new metrics or models with minimal manual customization.

### 3. Core Strategies for Effective Evaluation
To move beyond simple testing, AI teams should adopt these four strategies:

#### A. Employ Multiple Metrics
Avoid the "single metric trap." Combine various measurements to assess:
*   **Technical Quality:** Accuracy and task completion.
*   **Linguistic Quality:** Fluency, coherence, and relevance.

#### B. Incorporate Human Judgment
Since AI evaluation is inherently subjective, human oversight is essential:
*   Use **multiple human judges** to mitigate individual bias.
*   Conduct **inter-rater reliability checks** to ensure consistency.
*   Leverage **crowdsourcing** to gain a diverse range of perspectives.

#### C. Leverage Domain-Specific Data
Standardized benchmarks (like those for general knowledge) are often insufficient.
*   Incorporate industry-specific datasets to simulate actual real-world scenarios.
*   Ensure the model understands the specific jargon and context of your field (e.g., media, medical, or legal).

#### D. Adopt MLOps for Generative AI
Transition from manual checks to automated pipelines:
*   **Integrated Workflows:** Build evaluation directly into the fine-tuning process.
*   **Automated Triggers:** Ensure that every time a model is updated or fine-tuned, an evaluation job runs as the final step.
*   **Efficiency:** This eliminates manual bottlenecks and creates a continuous feedback loop for improvement.

### 4. Conclusion
There is no "perfect" single solution for LLM evaluation. Success relies on a combination of **automated MLOps**, **diverse human feedback**, and **ongoing adaptation** to new research and domain-specific challenges.

This summary details how **Vertex AI** implements model evaluation for Generative AI, focusing on the specific tools, pipelines, and evaluation paradigms provided by the platform.

## Generative AI Evaluation in Vertex AI

### 1. Core Evaluation Methods
Vertex AI provides two primary technical approaches to measuring model performance:

*   **Computation-based Method:** The traditional approach. It compares model outputs against a **Ground Truth** (human-labeled "gold standard") using mathematical formulas.
*   **Model-based Method:** A modern approach using a specifically tailored **LLM as a judge** (Arbiter) to assess the quality of another model's output.

### 2. Evaluation Pipeline Services
Vertex AI orchestrates the evaluation process through serverless pipelines. These are "end-to-end" solutions that handle response generation, service calls, and metric calculations.

*   **Best Use Cases:** 
    *   **Large-scale evaluations:** Where high volume offsets the initial startup latency.
    *   **Asynchronous workflows:** Where real-time results aren't required.
    *   **Automated MLOps:** Integrating evaluation into the broader model lifecycle.

### 3. Evaluation Paradigms: Pointwise vs. Pairwise
*   **Pointwise Evaluation:** Assesses the **absolute performance** of a single model.
    *   *Goal:* Establish a baseline and identify specific strengths/weaknesses for tuning.
    *   *Result:* Usually a numerical score.
*   **Pairwise Evaluation:** A **direct comparison** between two models (or two different prompts).
    *   *Goal:* Determine model selection or the impact of specific tuning/prompt engineering.
    *   *Result:* Identifies a "winner" or preferred model.

### 4. Deep Dive: Metric Categories

#### A. Computational Metrics (Quantitative)
Standardized and efficient, but limited by their inability to capture creative nuance.
*   **Lexicon-based:** Measures string similarity (e.g., Exact Match, ROUGE).
*   **Count-based:** Quantifies matches/mismatches (e.g., F1 Score, Accuracy).
*   **Embedding-based:** Compares numerical representations in a vector space to find semantic similarity.

#### B. Model-based Evaluation (Qualitative & Scalable)
Pioneered by Google Research, this uses "Arbiter models" to mimic human judgment.
*   **Auto Side-by-Side (Auto SxS):** Google's primary tool for on-demand assessment.
*   **Transparency Features:**
    *   **Explanations:** Uses "Chain of Thought" reasoning to explain *why* a score was given.
    *   **Confidence Scores:** A value (0 to 1) indicating how certain the Arbiter is, calculated via self-consistency decoding.

### 5. Implementation Strategy
To select the right approach in Vertex AI, practitioners should follow these steps:

1.  **Choose Paradigm:** Do you need an absolute score (Pointwise) or a comparison (Pairwise)?
2.  **Define the Task:** Vertex AI supports four broad tasks:
    *   Summarization
    *   Question Answering (QA)
    *   Tool Use
    *   General Text Generation
3.  **Prioritize Aspects:** Identify if Accuracy, Creativity, Safety, or Fluency is the most critical factor for the specific use case.
4.  **Configure Parameters:** Specify required inputs (e.g., temperature settings) and task-specific metrics (e.g., relevance or helpfulness).


> **Pro Tip:** Model-based evaluation is highly effective because it provides **contextual explanations** and **confidence scores**, moving beyond "black box" numerical outputs to provide actionable insights for model refinement.

This summary focuses on the technical implementation, requirements, and limitations of **Computation-Based Metrics** within Vertex AI for evaluating Large Language Models.


## Computation-Based Model Evaluation : Vertex AI

### 1. Overview of Computation-Based Metrics
Computation-based evaluation measures model performance by comparing the input prompt and output response pairs against a predefined "gold standard" (ground truth). 

*   **Alignment:** This method follows academic research standards and open benchmarks.
*   **Scope:** Supported for both **base** and **tuned** versions of Vertex AI text-based LLMs (e.g., PaLM).
*   **Focus:** Evaluation is centered on core comprehension capabilities rather than creative nuance.

### 2. Metrics by Task Type
Vertex AI provides specific metrics tailored to different machine learning tasks:

*   **Classification:** 
    *   Micro-F1
    *   Macro-F1
    *   Per-class F1
*   **Summarization:** 
    *   **ROUGE-L:** Measures the longest shared word sequence between the model output and the ground truth to assess if the summary captures the article's essence.
*   **Customization:** Additional metrics are available in the official documentation depending on the "supported task" category.

### 3. The Evaluation Workflow
To run a computation-based evaluation job, follow these three primary steps:

#### Step 1: Data Preparation
*   Create a dataset containing **prompt-ground truth pairs**.
*   **The Prompt:** Must include clear instructions and the necessary context.
*   **Ground Truth:** The "correct" answer used to calculate the final metrics.
*   **Volume:** A minimum of **10 examples** is required, though they must closely resemble real-world scenarios.

#### Step 2: Storage
*   Upload the prepared dataset to a **Google Cloud Storage (GCS)** bucket.

#### Step 3: Execution
*   Submit the evaluation job via the **Vertex AI Python SDK**, REST API, or Google Cloud Console.
*   Utilize the **Vertex AI Pipeline templates** for scalability.
*   **Parameters Required:** Dataset location, specific task type, output location, and the target model.

#### 4. Strengths and Limitations
While computation-based metrics are essential for benchmarking, they have specific trade-offs.

#### Strengths:
*   **Speed:** Provides rapid feedback on model performance.
*   **Objectivity:** Metrics are calculated mathematically, ensuring consistency across runs.
*   **Accessibility:** Easily triggered via SDKs or APIs for automated workflows.

#### Limitations:
*   **Lack of Nuance:** Aggregated metrics (like F1 or ROUGE) may indicate which model is "mathematically" better but fail to explain *why*.
*   **Human Alignment Gap:** These metrics do not necessarily reflect **human preference**. A model could have a high ROUGE score but produce a summary that feels robotic or unhelpful to a person.
*   **Scalability of Insight:** Manually reviewing pairs to determine preference is too time-consuming, creating a need for more automated "alignment" evaluations (e.g., Model-based evaluation).

This summary details the **Auto Side-by-Side (AutoSxS)** tool within Vertex AI, which provides an automated, "model-based" approach to comparing Large Language Models.

## Automatic Side-by-Side (AutoSxS) : Vertex AI

### 1. What is AutoSxS?
**AutoSxS** is an evaluation tool used for on-demand **A/B testing** of LLMs. It uses an "autorater" (a specialized judging model) to compare responses from two different models and determine which is superior based on specific criteria.

*   **The Autorater:** A specialized LLM trained to act as a surrogate for human judges.
*   **Transparency:** Unlike raw numerical scores, AutoSxS provides **explanations** at the input level, clarifying *why* one response was preferred over another.
*   **Use Cases:** Comparing two different model architectures (e.g., Gemini Pro vs. an open-source model) or comparing a fine-tuned model against its base version.

### 2. Key Capabilities
*   **Supported Models:** Any model in the Vertex AI Model Registry that supports batch prediction or pre-existing models with pre-generated predictions.
*   **Core Tasks:** Currently optimized for **Summarization** and **Question-Answering (QA)**.
*   **Predefined Criteria:** Evaluation includes adherence to instructions, groundedness, coherence, and coverage.

---

### 3. The Evaluation Dataset
To run AutoSxS, you must prepare a dataset (JSONL or BigQuery) where each row contains:
*   **ID Fields:** Unique identifiers for each example.
*   **Data Fields:** The original prompt and the context (e.g., the article to be summarized).
*   **Model Responses:** Pre-generated predictions from Model A and Model B.
*   **Volume Recommendation:** While 1 example is the minimum, **400–600 examples** are recommended for statistically significant aggregate metrics.

---

### 4. Execution Workflow
1.  **Prepare Data:** Format your prompts and responses into a JSONL file.
2.  **Upload:** Store the file in a **Google Cloud Storage** bucket or a **BigQuery** table.
3.  **Configure Parameters:** 
    *   `evaluation_dataset`: Path to the data.
    *   `id_columns`: e.g., "id" and "document".
    *   `task`: e.g., "summarization".
    *   `response_column_a` / `response_column_b`: The names of the columns containing model outputs.
4.  **Run Pipeline:** Use the **Vertex AI Python SDK** to submit the job using a provided Google Cloud pipeline template.

### 5. Interpreting Results
AutoSxS provides results in two primary formats:

#### A. The Judgment Table (Example-level)
For every single prompt, the table shows:
*   **The Winner:** Which model (A or B) performed better.
*   **Confidence Score:** A numeric value (0–1) indicating the autorater's certainty.
*   **Explanation:** String output using **Chain of Thought** reasoning (e.g., "Model B was favored due to higher coherence and better coverage").

#### B. Aggregated Metrics (Model-level)
*   **Win Rate:** The overall percentage of times the autorater preferred one model over the other.

---

### 6. Human Preference Alignment
AutoSxS allows you to include a "Human Preference" column in your dataset to validate the autorater.
*   **Benefit:** It calculates **alignment metrics** to show how closely the autorater’s choices match human decisions.
*   **Utility:** This builds trust in the automated system; if alignment is high, you can rely more heavily on the autorater for future evaluations at scale.

> **Key Takeaway:** AutoSxS replaces slow, expensive human A/B testing with a scalable, transparent, and objective model-based evaluation framework.
