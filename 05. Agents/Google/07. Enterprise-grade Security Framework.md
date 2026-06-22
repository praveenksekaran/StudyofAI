# Google Cloud's enterprise-grade security framework for agents

The ability of AI Agents to autonomously execute code and interact with external services poses security challenges related to prompt injection, unauthorized actions, and control over external systems. Thus, deploying AI Agents in a production environment requires a robust security strategy to protect your agent, its data, and the interactions it handles.  

In this lesson, you'll explore the key components of Google’s hybrid, defense-in-depth approach to AI Agent security.  

## Security challenges of how AI Agents work

AI Agents are systems designed to perceive their environment, make decisions, and take autonomous actions to achieve a user-defined goal. Unlike LLMs that only generate content, agents act by leveraging AI reasoning to interact with other systems like APIs, databases, or even web browsers.  

Select **+** to expand each section and learn more about the unique security risks of agents.  

#### Input, perception, and personalization
Agents begin by receiving input like typed commands, voice queries, or contextual data gathered from the environment. The input, which can be multimodal (text, image, audio), is processed and perceived by the agent and often transformed into a format the AI model can understand.

**Agents must know the difference between commands they should follow and other data they can't trust**. If an agent can't tell the difference, it opens the door to prompt injection attacks, where hidden, malicious instructions can take control of the system.

#### System instructions
The agent’s core model operates on a combined input in the form of a structured prompt. This prompt integrates predefined system instructions (which define the agent’s purpose, capabilities, and boundaries) with the specific user query and various data sources like agent memory or externally retrieved information.

**To prevent prompt injection attacks, it's important to separate safe system instructions  from unsafe user data or external information.**

#### Reasoning and planning
The processed input, combined with system instructions defining the agent’s purpose and capabilities, is fed into the core AI model. This model reasons about the user’s goal and develops a plan—often a sequence of steps involving information retrieval and tool usage—to achieve it.

**This planning can be iterative, refining the plan based on new information or tool feedback. However, each reasoning cycle introduces opportunities for flawed logic, divergence from intent, or hijacking by malicious data, potentially compounding issues.**

#### Orchestration and action execution (tool use)
To execute its plan, the agent interacts with external systems or resources via “tools” or “actions.” These could be through APIs for sending emails, querying databases, accessing file systems or controlling smart devices, or even interacting with web browser elements.

Each tool grants the agent specific powers and uncontrolled access to powerful actions, such as deleting files or making purchases, is highly risky if the planning phase is compromised.

**Secure orchestration requires robust authentication and authorization for tool use, ensuring the agent has appropriately constrained permissions (reduced privilege) for the task at hand.**

#### Agent memory
Many agents maintain some form of memory to retain context across interactions, store learned user preferences, or remember facts from previous tasks.

**If malicious data containing a prompt injection is processed and stored in memory (for example, as a “fact” summarized from a malicious document), it could influence the agent’s behavior in future, unrelated interactions.**

#### Response rendering
If the application renders agent output without proper sanitization or escaping based on content type, vulnerabilities like Cross-Site Scripting (XSS) or data exfiltration (from maliciously crafted URLs in image tags, for example) can occur.


### Key risks associated with AI Agents

The inherent design of AI Agents, combined with their powerful capabilities, exposes users to two major risks:  

- 1. **Rogue actions:** Unintended, harmful, or policy-violating behaviors. This often stems from prompt injection, where malicious data tricks the agent into executing attacker commands using the user's privileges.  

- 2. **Sensitive data disclosure:** The agent improperly revealing private or confidential information. Attackers exploit actions (data exfiltration) or output generation to trick the agent into leaking secrets via a URL or an insecurely rendered response.  

## Core principles for agent security

Google advocates for three fundamental principles to guide secure agent development.  

- **Principle 1: Agents must have well-defined human controllers.** This ensures accountability, user control, and prevents agents from acting autonomously in critical situations without clear human oversight or attribution.  

- **Principle 2: Agent powers must have limitations.** Enforcing appropriate, dynamically limited privileges, ensures agents have only the capabilities and permissions necessary for their intended purpose and cannot escalate privileges inappropriately.  

- **Principle 3: Agent actions and planning must be observable.** Effective observability requires transparency and auditability through robust logging of inputs, reasoning, actions, and outputs, enabling security decisions and user understanding.  

## Google’s hybrid defense-in-depth approach

Since LLMs are non-deterministic and susceptible to manipulation, neither purely rule-based systems nor purely AI-based judgment are sufficient alone. Google employs a **hybrid defense-in-depth strategy which combines traditional, deterministic measures with reasoning-based defense strategies**.  

- 1. TRADITIONAL, DETERMINISTIC MEASURES
This layer uses dependable security mechanisms, providing runtime policy enforcement that operates outside the AI model's reasoning.
  - Function: Intercepts agent actions (tool use) before execution.
  - Action: Evaluates the request against predefined, auditable rules (e.g., "Block any purchase action over $500," or "Require user confirmation for high-risk actions").
  - Benefit: Provides reliable, predictable hard limits and confines the potential impact of an agent malfunction.  

- 2. REASONING-BASED DEFENSE STRATEGIES
This layer uses AI models to evaluate inputs, outputs, or internal reasoning for potential risks, making the agent core more resilient, providing model hardening and guardrails.
  - Function: Makes the LLM resistant to manipulation.
  - Techniques:
    - Adversarial training: Exposing the model to various prompt injection attacks to train it on the desired safe response (e.g., ignore the malicious instructions).
    - Specialized guard models (classifiers): Smaller AI models trained to function as security analysts, examining inputs or outputs for subtle signs of attack that static rules might miss.
  - Benefit: Handles dynamic and novel malicious patterns, increasing the attacker's difficulty and cost. 

This hybrid, layered approach creates robust boundaries, mitigating the risk of harmful outcomes while preserving the agent's utility.  

## Securing AI applications with Model Armor

**Model Armor** is a Google Cloud service designed to enhance the security and safety of AI applications. It is configured for integration with Google Cloud services like Vertex AI and Gemini Enterprise.  

By proactively screening large language model (LLM) prompts (inputs) and responses (outputs), Model Armor protects the agents built with Vertex AI Agent Builder against critical risks.  

### Prompt injection and jailbreak attacks

Model Armor proactively identifies and blocks malicious inputs designed to manipulate the LLM's system instructions or behavior, ensuring the AI agent operates as intended.  

#### Prompt injection risk mitigation
- Risk: Attempting to override system instructions or alter the model's objective.
- How you can mitigate this risk: Implement a guardrail to detect and filter prompts attempting to bypass, ignore, or rewrite the model's safety and system instructions.
- How Model Armor can help you: Model Armor uses a multi-layered defense system combining heuristic rules (pattern-based detection), machine learning models, and contextual analysis to detect the structure and intent of manipulative prompts. It automatically blocks, sanitizes, or flags these inputs before they reach the Vertex AI model, preventing the LLM from executing the unintended command.

#### Jailbreak attack risk mitigation
- Risk: Techniques used to bypass the safety controls and restrictions built into the LLM.
- How you can mitigate this risk:  Use an input guardrail with advanced detection (ML models, contextual analysis) to identify and block sophisticated jailbreaking attempts before the core LLM processes them.
- How Model Armor can help you: Model Armor specifically targets and mitigates sophisticated jailbreak attempts by continuously monitoring the model's input-output chain. It uses a specialized LLM-based detector to identify patterns indicative of known jailbreak techniques (e.g., role-playing, preambles) and intercepts the malicious query, protecting the integrity of the AI agent's safety guardrails.

### Sensitive data leakage

Model Armor mitigates the risk of leaking sensitive data like intellectual property (IP) and personally identifiable information (PII) in prompts or responses. Select each tab to learn more about how Model Armor can help mitigate these security risks.  

- SENSITIVE DATA IN PROMPTS
 - Risk: User input (prompt) containing sensitive data that should not be processed or stored by the LLM.
 - How you can mitigate this risk: Implement an input filtering mechanism to scan and remove or obscure sensitive data.
 - How Model Armor can help you: Model Armor uses the Cloud Data Loss Prevention service to scan all incoming user prompts for defined sensitive data types (e.g., credit card numbers and email addresses). It can be configured to automatically mask, tokenize, or redact this data in real-time before it's processed by the LLM, protecting the underlying data from exposure during the session.

- SENSITIVE DATA IN RESPONSES
  - Risk: An LLM may inadvertently generate and reveal sensitive data in its response, which is known as unintended disclosure.
  - How you can mitigate this risk: Implement an output filtering mechanism to scan the LLM's generated response for sensitive data and mask or redact the data before the final response is delivered to the user.
  - How Model Armor can help you: Model Armor scans the LLM-generated responses before they are returned to the user. If the response contains sensitive data, Model Armor blocks the response or redacts the sensitive data, ensuring the final output presented to the end-user adheres to compliance and privacy standards.  

### Content safety

**The risk:** LLM outputs can be unpredictable and potentially harmful, so they need to be reviewed.  

**How you can mitigate this risk:** To utilize this technology safely and responsibly, it is also important to consider other risks specific to your use case, users, and business context in addition to built-in technical safeguards.  

We recommend taking the following steps:  

1. Assess your application's security risks.
2. Perform safety testing appropriate to your use case.
3. Configure safety filters if required.
4. Solicit user feedback and monitor content.  

You can use existing tooling for safety filtering. The Gemini API, for example, provides adjustable controls over content generation across harm dimensions like harassment, hate speech, and dangerous content through its [safety settings (opens in a new tab)](https://ai.google.dev/gemini-api/docs/safety-settings).  

**How Model Armor mitigates this risk:** Model Armor serves as another layer of defense to filter content against safety categories, such as dangerous content, ensuring responsible AI outputs.  
Model Armor leverages a pre-trained safety classifier which analyzes the semantic meaning and context of both the prompt and the response against a configurable set of safety categories and thresholds. If a response from the model exceeds the risk threshold for a given category, Model Armor automatically blocks the response or returns a customized safety message.  


# Authentication and authorization for AI Agents

Imagine your AI Agent is a new employee. Before they can start working (e.g., retrieving data, calling an external API, or saving a file), they need proper credentials and permissions. If you give them too much access, it creates a security risk. If you give them too little, they can't do their job.  

This lesson explores verifying credentials and managing permissions for AI Agents in Google Cloud to control what they can and cannot do after deployment.  

## Authentication vs. authorization

The security of AI Agents in Google Cloud relies on a dual-faceted access control system, and every access request made by your agent involves these two distinct security steps:  

1. **Authentication (AuthN):** Who are you? This is the process of verifying the agent's identity using its credentials (the Service Account, API Key, or OAuth ID). Once verified, the agent is considered a known "Principal."  

2. **Authorization (AuthZ):** What are you allowed to do? This is the process of checking the permissions (Roles) granted to that Principal. IAM determines if the verified agent is authorized to perform the requested action (e.g., storage.objects.get).  

Put simply, authentication gets the agent's foot in the door, but authorization then tells the agent what rooms it can enter.  

## Service accounts

Authorization for Google Cloud is primarily handled by Identity and Access Management (IAM). IAM offers granular control by principal and by resource. When an agent is deployed via the Vertex AI Agent Engine, it needs a core identity (the principal) that Google Cloud can track and grant roles to. A deployed agent runs using either a default or custom service account.  

The default identity is called AI Platform Reasoning Engine Service Agent and it is a Google-managed service account automatically created for your project. It is automatically assigned the roles/aiplatform.reasoningEngineServiceAgent role, which contains the standard permissions necessary for basic agent operation.  

You can also choose to deploy an agent with a custom service account that you created and manage. A custom account allows for much tighter control using the principle of least privilege. You only grant the specific roles that your agent needs, reducing the attack surface.  

## Authentication methods

AI Agents often need to access different resources, which may require different methods of authentication.  

| Authentication method | Use case | Description |
|---|---|---|
| **Service account** | Accessing Google Cloud resources (e.g., Cloud Storage, BigQuery, other Vertex AI services). | The deployed agent's core service account identity is used. The account must have the necessary IAM roles granted at the project or resource level. |
| **API keys** | Sending requests to external endpoints like APIs. | Used when the external service requires an API key, which must be secured, often via Secret Manager. |
| **OAuth Client ID** | Handling user accounts, login, or authorization for the agent's end-users. | This method requires the agent to request and receive explicit consent from the end-user (e.g., "Allow Agent X to access your calendar"). |

## Granting and revoking permissions

Since authorization relies on the roles assigned to the agent's service account principal, managing access is done entirely through assigning or removing IAM roles. This process ensures the agent only has the necessary permissions.  

You have two primary options for managing these assignments. You can use the Google Cloud console for one-off changes, but the gcloud Command Line Interface (CLI) is recommended for programmatic or bulk role management.

# Deploying and managing AI Agents in production

Successfully moving an AI Agent from a prototype to a production environment requires a robust, scalable, and secure infrastructure. This lesson explores Google Cloud's comprehensive offerings and tools to manage the full lifecycle of your agents, from initial deployment to continuous monitoring and scaling.  

## Deployment and infrastructure

The choice of hosting platform depends on your agent's complexity, language, and required level of management. 

#### Agent Engine

For agents built using the Agent Development Kit (ADK), Vertex AI Agent Engine provides the most streamlined, managed path to production through the following features:  

- **Managed service:** This is a fully managed runtime environment specifically optimized for agent workloads. It handles infrastructure, scaling, and integrated memory/session management automatically.  
- **Simplified deployment:** Developers can typically deploy their agents with a single Command Line Interface (CLI) action, significantly accelerating the path from development to live service.  
- **Security and interoperability:** It natively supports standards like the Agent-to-Agent (A2A) protocol for secure collaboration between different agents and includes built-in security features.  

Ultimately, these capabilities enable Gemini Enterprise to utilize Vertex AI Agent Engine as its managed runtime and governance layer, enabling organizations to securely deploy, scale, and manage all custom and pre-built AI agents for automated workflows.  

#### Cloud Run

For highly customized agents or those requiring a specific execution environment, Cloud Run is the recommended serverless container platform.  

- **Flexibility:** It allows you to package your agent and all its dependencies into a container, offering maximum control over the runtime environment (including CPU/GPU allocation).  
- **Autoscaling:** Cloud Run scales automatically and rapidly, often down to zero when not in use (pay-per-use pricing), making it highly cost-efficient for variable traffic.  
- **Exposition:** Cloud Run is ideal for exposing agents via multiple mechanisms—as a web UI, a REST API endpoint, or through A2A communication.  

#### Google Kubernetes Engine (GKE)

GKE is a container orchestration platform that's ideal for running many instances of an AI agent simultaneously. GKE provides the necessary control and scaling for complex AI workloads:  

- **Specialized hardware support:** GKE is designed for deploying sophisticated, high-performance AI agents and their underlying models. It's ideal for users who require maximum control, specialized hardware, and a production-grade orchestration ecosystem.  
- **Full customization and orchestration:** GKE offers the entire Kubernetes feature set, giving you fine-grained control over networking, storage, security, and the distribution of complex, multi-component agent architectures.  
- **Advanced scaling:** Beyond basic scaling, GKE provides AI-aware autoscaling features like the GKE Inference Gateway and custom Horizontal Pod Autoscalers (HPA). This allows for optimized resource use, lower tail latency, and high throughput for demanding, variable traffic loads.  
- **Production Readiness:** GKE ensures production stability through integrated security, detailed observability, and compatibility with MLOps pipelines.  

#### App Engine

Agents built with the Conversational Agents platform are often deployed and hosted using App Engine for scalable, serverless backend processing.  

- **Flexibility:** App Engine provides broad support for modern web service code. The service's runtime environment, including specifying the exact language version (e.g., go116 or Python 3.10), is easily defined in the app.yaml configuration file.  
- **Autoscaling:** App Engine has reliable and robust automatic scaling which helps to manage traffic fluctuations without manual intervention.  
- **Exposition:** The agent's user interface comes from Dialogflow Messenger. By embedding its code into a simple webpage, users can start interacting with the agent right away.  

#### Compute Engine

Compute Engine provides Infrastructure-as-a-Service (IaaS), offering highly customizable virtual machines (VMs). It represents the lowest level of abstraction among Google Cloud compute options, giving you the most control over the underlying environment.  

- **Customization:** You can use Compute Engine when your AI agent has rigid, highly specific operating system, kernel, or networking requirements that cannot be met by managed services like Cloud Run or GKE.  
- **Legacy workload support:** It is the ideal choice for migrating existing AI systems or legacy applications that require a specific OS or licensed software to run alongside the agent.  
- **Persistent services:** Compute Engine VMs are well-suited for agents that are long-running, stateful, and require continuous, predictable resource allocation rather than the elastic, scale-to-zero nature of serverless platforms.  
- **Management overhead:** The trade-off for the customization and control is full responsibility for managing the VM, including the operating system, patching, scaling, security, and networking configuration. It requires the highest operational overhead.  

## Security and access control

Production environments require strict control over who can access the agent, and what external systems the agent can interact with.  

| Security focus              | Google Cloud tool/feature                 | Purpose in agent security                                                                                                                                                                                                 |
|----------------------------|--------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Identity and authorization** | **Identity and Access Management (IAM)** | Ensures only authorized human users can interact with the agent (e.g., via a secured web UI) and grants the agent itself a service account with the minimum necessary permissions for its tools (principle of least privilege). |
| **Data perimeter**         | **VPC Service Controls**                  | Creates a secure, impenetrable perimeter around sensitive Google Cloud resources (like data stores and vector databases) to prevent unauthorized movement of data, directly mitigating the risk of data exfiltration.     |
| **Tool permissions**       | **Scoped credentials and service accounts** | The agent's identity is used to call external APIs, ensuring that its access to a database or a file system is always constrained to exactly what is needed for the task at hand.                                          |

## Monitoring

Observability is important for diagnosing performance bottlenecks, auditing security events, and tracking business value.  

- **Logging:** Cloud Logging is used to capture comprehensive, structured logs of every agent interaction, tool invocation, and error. This supports the agent observability principle.  
- **Tracing:** Cloud Trace helps visualize the end-to-end latency of a conversation, breaking down time spent in LLM calls, tool executions (webhooks), and memory retrieval, allowing you to quickly find and fix bottlenecks.  
- **Metrics:** Cloud Monitoring tracks key operational metrics such as QPS (queries per second), error rates, and average latency.  
- **Agent-specific metrics:** Platforms like Conversational Agents provide specialized analytics for AI performance, tracking metrics like escalation rates, conversation outcomes, and tool failure rates.  

## Lifecycle management

Agents are constantly evolving and a clear process for updates ensures reliability. 

- **VERSIONING**
  Use the built-in versioning capabilities of services like Vertex AI Agent Engine or Cloud Run to snapshot your agent's code, models, and configuration at stable points.
  
- **ENVIRONMENTS**
  Deploy new agent versions to staging or test environments first, allowing your internal teams and red teams to validate behavior and security before pushing to production.
  
- **ROLLBACKS**
  Maintain a quick, reliable path to roll back to a previously stable version in case a new deployment introduces unforeseen errors or security regressions. This is critical due to the non-deterministic nature of AI model behavior.
