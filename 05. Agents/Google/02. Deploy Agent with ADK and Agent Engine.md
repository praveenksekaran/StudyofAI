# Introduction

Google’s **Agent Development Kit (ADK)** is a Python SDK for building, evaluating, and deploying multi‑agent systems on a fully managed Agent Engine runtime, without needing AI expertise.[1]

## Why AI agents and where they help

- By 2028, over 33% of enterprise software will include AI agents and they will automate about 15% of daily work decisions, transforming operations across industries.[1]
- AI agents already assist data scientists (datasets, anomaly detection, visualizations), employees (onboarding, underwriting, claims, returns), research analysts (summaries, recall, hypotheses), and developers (design, coding, bug logging, tickets, debugging).[1]

## Agent-building options on Google Cloud

- Approaches range from custom low-level builds (Vertex + function calling, Gen AI SDK, LangChain, GenKit) to higher-level, out-of-the-box solutions (Agentspace, Conversational Agents).[1]
- Google Cloud highlights four main paths: Customer Engagement Suite (Conversational Agents), Agentspace/Gemini Enterprise, custom agents with Agent Engine or open-source frameworks, and using ADK with Agent Engine.[1]

## When to choose each path

- Use **Customer Engagement Suite** for external-facing conversational agents integrated with telephony, chat, and human support workflows.[1]
- Use **Agentspace or Gemini Enterprise** for internal enterprise search and knowledge exchange across drives, mail, chat, tickets, and databases with AI assistant support.[1]
- Use low-level SDKs or open-source frameworks when you need full control and are willing to manage infrastructure and hosting yourself.[1]

## What ADK provides

- ADK targets developers who want code-centric agent development, with strong support for multi-agent communication and state, but without needing AI expertise.[1]
- It is a client-side Python SDK for building, managing, evaluating, and deploying AI-powered agents and complex multi-agent systems.[1]

## Multi-agent capabilities and tools

- ADK supports hierarchical multi-agent systems where a parent agent routes work to specialized subagents to complete different tasks.[1]
- Agents can use tools beyond conversation—calling external APIs, searching information, running code, or invoking other services to act in the real world.[1]

## Open ecosystem and integration

- ADK promotes an open ecosystem by allowing easy integration and reuse of tools from other frameworks such as LangChain and CrewAI.[1]
- This lets teams leverage existing community tools and prior investments instead of rebuilding everything from scratch.[1]

## Evaluation, debugging, and state

- ADK simplifies evaluation and provides a local development UI to inspect, debug, and iterate on agents and multi-agent flows.[1]
- It offers callbacks for hooking into various stages of a flow, session memory for long-term user context across sessions, and artifact storage to support collaboration on documents.[1]

## Deployment with Agent Engine

- ADK integrates with **Agent Engine**, which offers fully managed, autoscaling infrastructure for agent workloads.[1]
- This removes infrastructure and scaling concerns so developers can focus on agent logic and interactions.[1]

[1](https://www.skills.google/paths/3273/course_templates/1275/video/606585)

# Develop agents with ADK

<img width="583" height="366" alt="image" src="https://github.com/user-attachments/assets/30db041d-651c-4402-bec7-e3580d0f7a6a" />

Google **Agent Development Kit (ADK)** provides core primitives (agents, tools, workflows, models, runtime) plus rich dev tooling to build, run, and debug production-grade AI agents.[1]

## Core ADK concepts

- An **Agent** is the fundamental worker unit that uses LLMs to reason, control workflows, and delegate sub-tasks to other agents or tools, enabling modular multi-agent systems.[1]
- ADK supports bi-directional streaming (text and audio) and integrates with capabilities like Gemini Live API for real-time interactive experiences via simple configuration.[1]

## Artifacts, tools, and developer tooling

- **Artifact Management** lets agents save, load, and version artifacts (files, images, reports) tied to a user or session during execution.[1]
- A rich tool ecosystem supports custom functions, treating other agents as tools, built-in code execution, external APIs/data sources, and long-running asynchronous tools.[1]
- Integrated dev tooling includes a CLI and Web UI to run agents, inspect steps, debug interactions, and visualize agent definitions locally.[1]

## Sessions, events, memory, and orchestration

- **Session** represents a single conversation; **Events** are the units of communication (user messages, agent replies, tool calls) forming history; **State** is the agent’s working memory within that session.[1]
- **Memory** stores long-term user information across sessions, distinct from per-session State, enabling persistent personalization.[1]
- Flexible orchestration combines workflow agents and LLM-driven routing, using a **Runner** to manage execution flow, handle Events, and coordinate backend services.[1]

## Evaluation, code execution, and observability

- Built-in **Agent evaluation** lets you create multi-turn eval datasets and run evaluations via CLI or UI to measure quality and guide improvements.[1]
- **Code execution** tools allow agents to generate and run code for complex calculations or actions, while **callbacks** inject custom logic at key points for logging, checks, or behavior changes.[1]
- Cloud Trace integration collects traces from Agent Engine deployments to debug latency and interactions between LLM agents and tools.[1]

## Models and agent components

- ADK works with LLMs like **Gemini** and Claude, and via a Base LLM interface can integrate other, including open-source or fine-tuned, models.[1]
- An agent consists of four main components: **models** (reasoning and response), **tools** (data/actions via APIs/services), **orchestration** (task steps, planning, state, memory), and **runtime** (executes workflows when user queries arrive).[1]

## Deployment with Agent Engine

- ADK agents deploy to **Agent Engine**, a fully managed Google Cloud service that handles scaling and infrastructure for production workloads.[1]
- This lets developers focus on agent logic and workflows rather than provisioning and managing compute.[1]

<img width="874" height="402" alt="image" src="https://github.com/user-attachments/assets/d4560fd8-050f-46dc-8d26-41e25989a93e" />

[1](https://www.skills.google/paths/3273/course_templates/1275/video/606587)


# Configure ADK

## Installation 

```
# Create virtual env
python3 -m venv .adk

# Activate virtual environmnet
.adk\Scripts\activate

# Deactivate env
deactivate

# Install google ADK
pip install google-adk # Python 3.9+

#verify Installation
pip show google-adk or pip list | grep google-adk
```

## Directory structure 

<img width="308" height="282" alt="image" src="https://github.com/user-attachments/assets/9336c05e-e0fa-4438-a842-a91fadb6c3b2" />

There is a directory structure that should be maintained to organize your agents and tools. Each agent gets a directory, 
and each agent directory should contain an init file and an agent dot py file.

<img width="413" height="148" alt="image" src="https://github.com/user-attachments/assets/4e99aa7d-3232-4dd3-b0b5-280ac11fdd72" />

The use of dot env files in agent directories helps provide configurations for each agent.
Dot env files tell ADK agents whether to authorize through the Gemini API or through Vertex AI.

## Basic Agent code

<img width="894" height="374" alt="image" src="https://github.com/user-attachments/assets/d232d1ba-9909-47e9-bc14-1500276b1ee2" />

## Interact with Agents 
There are 4 ways 

1. Web UI
   With Web UI, you can interact with your agent through a user-friendly web browser. Web UI is a good option for visual interaction while
   developing your agent and monitoring agent behavior. It should only be used for local testing, it is not suitable for a production environment.
   To get started with Web UI, open your terminal, and use cd to navigate to the directory containing your agent folder.
   ```
   adk web
   ```
   
2. CLI
   With the CLI, you can use terminal commands to interact directly with your agent. The CLI is a good option for quick tasks, scripting, automation,
   and developers comfortable with terminal commands. This should also only be used for local testing, it is not suitable for a production environment.
   To get started with the CLI, open your terminal, and use cd to navigate to the directory containing your agent folder.
   Next, run this command to start the agent, and then you can interact with it directly in the terminal.
   ```
   adk run <agnets folder name>
   ```
   
3. API Server
   Run your agent as a REST API, allowing other applications to communicate with it. Running an API server is a good option for integration with other
    applications, building services that use the agent, and remote access to the agent. This approach can be used for production environments.
   To get started with an API server, open your terminal, and use cd to navigate to the directory containing your agent folder.
   Next, run this command to start a local API server, using Flask, on port 8000
   ```
   adk api_server <my_google_search_agent> # replace with folder name
   ```
   You can then interact with your agent through REST API calls.
   
4. Programmatic Server 
   The Programmatic Interface allows you to integrate ADK directly into your Python applications, or interactive notebooks (like Jupyter and Colab).
   Unlike the CLI, Web UI, and API server, you don't need the specific project structure, as previously described. Instead you’ll be using a Session and Runner.
   You can define and interact with your agent within the same file or notebook cell.  The programmatic interface provides deep integration within applications,
   custom workflows, notebooks, and fine-grained control over agent execution. This approach can be used for production environments.
   The Programmatic Interface requires that you handle setting up memory, including the in-memory session service and the in-memory artifact service.
   You also need to create a new session, prepare content, such as the user query, for the agent,  use a runner to execute the agents logic,
   and process the event stream to get the final response.

     ```
     # Create in-memory session for session and Artifact managment
     session_service - InMemorySessionService()
     artifact_service = InMemoryService()

     # Create new session
     session = session_service.create(app_name=AGENT_APP_NAME, user_id='user',)

     # Create a content object representing the users query
     query = "Hi, How are you >"
     content = types.content(role='user' , parts=[types.Part(text=query)])
     ```

   To create a programmatic run, you create in-memory session and artifact services.
   - The Session Service handles sessions, which store the conversational history, the agent's internal state, including variables, and other data
     related to a specific interaction. It's ephemeral in this in-memory implementation.
   - The Memory Artifact stores files, or data generated or used by the agent. This could be text files, images, or any other kind of data.
     Like sessions, these are lost when the program ends in the in-memory version.
   - A new session is used to create a track of each conversation. Content is prepared, encapsulating a user’s text query into Parts, and into a
     Content object, from the Google Gen AI dot types package.
   - A “Role” is also added, defining who is sending the message.
  
   ## Running an Agent
   Now let’s discuss what the Runner does. Initially, it orchestrates agent execution. The Runner takes a user's query, the agent's definition
   (including its instructions, model, and tools),    and a session context. It then manages the process of sending the query to the LLM, interpreting
   the LLM’s response, calling any necessary tools, and updating the agent’s state. It also manages sessions and artifacts. The Runner works in conjunction
   with the Session Service and Artifact Service, to maintain the context of the conversation (or the session) and manages any files or data
   (such as artifacts) that the agent uses or creates. When an agent runs, it generates a response consisting of one or more events. These events can include
   things like: The LLM’s response, A tool being called, The result of a tool call, Or the agent's final response.
   You can loop through the events returned in the event stream and print or use the responses as desired.

   ```
   # create a runner object to manage the interaction with the agent
   runner = Runner (AGENT_APP_NAME, agent, artifact_service=artifact_service, session_service=session_service)

   # Run the interaction with the agent and get a stream of events
   events - runner.run(session_id = session.id, user_id="user001", new_message=content)

   # Lop through the events returned by the runner
   final_response = None

   for _,event in enumerate(events):
     is_final_response = event.is_final_response()
     if is_final_respoinse:
       final_response = event.content.parts[0].text
   printf(f'Final Response : {final_response}')
   ```

   # Build Multi-agent systems with ADK

   ## Basics 

### Agent hierarchy and transfer rules

- Agents are organized in a tree; they can hand off only to sub‑agents, back to their parent, or to peers that share the same parent (peer transfer can be disabled per agent).[1]
- This constrained routing avoids accidentally calling unrelated agents and ensures only the lookup agent relevant to that part of the conversation is invoked.[1]
- <img width="891" height="398" alt="image" src="https://github.com/user-attachments/assets/12305569-509b-4eb8-b2c6-ceedb8ec0c63" />


### LLM agents vs workflow agents

- LLM‑based agents are the **brains**: they use large language models to understand natural language, make decisions, use tools, and typically alternate turns with the user.[1]
- Workflow agents are not powered by LLMs; they act as directors that route context between sub‑agents, enforcing sequence and conditions for structured processes.[1]

### Deterministic workflows

- Workflow agents usually produce deterministic execution: for a given input and configuration, the same sequence of agents will always run.[1]
- This predictability is crucial for processes where consistent, reliable behavior is required (for example, compliance‑sensitive flows).[1]

### Types of workflow agents

- **Sequential Agent:** Runs a fixed list of agents in order, such as “Order Validation → Inventory Check → Payment Processing → Order Confirmation” for a new order.[1]
  <img width="510" height="211" alt="image" src="https://github.com/user-attachments/assets/798941d9-8499-4fae-bef4-a695faf2d285" />

- **Loop Agent:** Repeats a set of agents until a condition is met, ideal for iterative research, monitoring, or negotiation cycles.[1]
 <img width="508" height="342" alt="image" src="https://github.com/user-attachments/assets/b6169653-04f8-457d-b38a-a75b4d29b2d4" />

- **Parallel Agent:** Executes multiple agents simultaneously to speed up independent subtasks, like generating regional reports in parallel when no shared state is needed.[1]
  <img width="450" height="368" alt="image" src="https://github.com/user-attachments/assets/922f3f9f-fbf3-45ac-949f-60949c3970e4" />

- **Custom workflow agents:**
Custom workflow agents let you define arbitrary orchestration logic beyond Sequential, Loop, and Parallel patterns.[1]
They support complex workflows, stateful interactions, and custom business rules tailored to specific enterprise scenarios.[1]
<img width="844" height="432" alt="image" src="https://github.com/user-attachments/assets/469961e5-0001-4d03-9cf4-63e0e577cc29" />

[1](https://www.skills.google/paths/3273/course_templates/1275/video/606592)

## Callbacks
Callbacks in ADK are **interception hooks** that let you observe, modify, or short‑circuit an agent’s execution at specific lifecycle points.[1]

### What callbacks are

- Python functions registered on an agent; the framework calls them automatically at predefined stages of a run.[1]
- Available on any agent inheriting from Base Agent (LLM Agent, Sequential, Parallel, Loop) with specialized hooks for LLM and tool interactions.[1]

### Agent lifecycle callbacks

| Callback            | When it runs                                   | Typical uses                                                                                  | Can skip main run? |
|---------------------|-----------------------------------------------|-----------------------------------------------------------------------------------------------|--------------------|
| **Before Agent**    | Immediately before `run` executes             | Setup resources, validate session state, log entry, modify state, return cached/default reply | Yes (by returning response) [1] |
| **After Agent**     | After `run` completes successfully            | Cleanup, post‑validation, log completion, modify final state/output                           | No (only post‑process) [1] |

- After Agent does not run if Before Agent already returned content or if execution ended early via “end invocation”.[1]

### LLM interaction callbacks (LLM Agent only)

| Callback            | Before/After          | Key powers                                                                                                             | Skip LLM call? |
|---------------------|-----------------------|------------------------------------------------------------------------------------------------------------------------|----------------|
| **Before Model**    | Before request to LLM | Inspect/modify request, add dynamic instructions, few‑shot examples, change model config, guardrails, request caching | Yes (return LLM Response) [1] |
| **After Model**     | After LLM response    | Inspect/modify raw response, log outputs, censor sensitive text, parse structured data into state, handle errors      | No (post‑process only) [1] |

- Returning an LLM Response from Before Model makes the framework treat it as if it came from the model.[1]

### Tool execution callbacks (LLM Agent only)

| Callback           | Before/After                 | Use cases                                                                                       | Skip tool run? |
|--------------------|------------------------------|--------------------------------------------------------------------------------------------------|----------------|
| **Before Tool**    | Before a tool’s `run_async`  | Inspect/modify tool args, auth checks, log usage, tool‑level caching                            | Yes (return dict) [1] |
| **After Tool**     | After `run_async` completes  | Inspect/modify tool result, post‑process/format, save parts to session state, filter outputs    | No for execution; can replace result with new dict [1] |

- If Before Tool returns a dictionary, the tool is not called and that dictionary becomes the tool result.[1]
- If After Tool returns a new dictionary, it replaces the original tool response seen by the LLM.[1]

### “Before vs After” behavior summary

- **Before callbacks (Agent/Model/Tool):**  
  - Intercept *inputs* to a stage.  
  - Can implement validation, guardrails, caching, and may completely skip the underlying run by returning a replacement result.[1]

- **After callbacks (Agent/Model/Tool):**  
  - Intercept *outputs* from a stage.  
  - Used for logging, cleanup, censorship, transformation, or storing derived data; they cannot re‑run the underlying step, only modify its result.[1]

[1](https://www.skills.google/paths/3273/course_templates/1275/video/606593)

# Deploy agents to Agent Engine 
Agent Engine is a **fully managed runtime** on Vertex AI for deploying, scaling, and operating ADK and other framework-based agents in production.[1]

## What Agent Engine provides

- Secure, scalable deployment of Gen AI agents with APIs for managing and querying agents.[1]
- Handles infrastructure concerns like authentication, VPC‑SC, and reliability so you focus on agent logic.[1]

## Key capabilities

- **Comprehensive agent management:** Session and memory management, plus UI to list, inspect, and converse with agents.[1]
- **Monitoring and analytics:** Built‑in tools to observe behavior and performance of deployed agents.[1]
- **Quality and evaluation:** Evaluation tools and an Example Store to measure quality and iteratively improve agents.[1]

## Framework support

| Aspect            | Agent Engine behavior                                                                 |
|-------------------|----------------------------------------------------------------------------------------|
| Framework support | Framework‑agnostic: supports ADK, Vertex AI Agent Framework, and OSS (LangGraph, LangChain, CrewAI).[1] |
| Role in stack     | Keystone that connects development, observation, evaluation, and deployment workflows.[1] |

## Typical lifecycle with Agent Engine

1. **Develop & test:** Define models, tools, and agents locally; monitor, evaluate, and test behavior.[1]
2. **Register & deploy:** Register agents and deploy them to a remote app in Agent Engine.[1]
3. **Query & operate:** Query the deployed agents from your app and use monitoring/analytics for ongoing operations.[1]

## How Agent Engine reduces friction

- Addresses common blockers like timeouts, model selection, security services, CI/CD integration, databases, and analytics via an end‑to‑end managed solution.[1]
- Provides the shortest path from prototype to production by unifying deployment, evaluation, and operations in one platform.[1]

[1](https://www.skills.google/paths/3273/course_templates/1275/video/606596)

# Extend Agents with MCP and A2A
To Do : Labs 

# Evaluate and Test ADK Agents SDK 
Evaluation in ADK focuses on **both final answers and the decision-making trajectory**, using automated tests and datasets to bridge PoC to production.[1]

## What is evaluated

- Final response: quality, relevance, and correctness of the agent’s answer vs a reference response using a ROUGE-based response_match_score (default 0.8).[1]
- Trajectory & tool use: the sequence of actions (tool calls, decisions) vs an expected “ideal” trajectory, scored by tool_trajectory_avg_score (default 1.0 for 100% match).[1]

## Trajectory metrics and when to use them

| Metric type      | What it checks                                                     | Typical use cases                                                | Strictness |
|------------------|--------------------------------------------------------------------|------------------------------------------------------------------|-----------|
| Exact match      | Predicted trajectory exactly equals ideal trajectory.[1]     | High‑stakes flows where every step must match.                   | Very high |
| In‑order match   | Correct actions in correct order; extra actions allowed.[1]  | Flows needing order correctness but tolerating extra steps.      | High      |
| Any‑order match  | Correct actions in any order; extra actions allowed.[1]      | Flexible workflows where order is not critical.[1]         | Medium    |
| Precision        | How many predicted actions are relevant/correct.[1]          | Minimizing unnecessary or harmful actions.                       | Medium    |
| Recall           | How many essential actions are captured.[1]                  | Ensuring no critical step is missed.[1]                    | Medium    |
| Single‑tool use  | Whether a specific action/tool was used.[1]                  | Validating inclusion of a key tool call.                         | Targeted  |

- Choose stricter metrics (Exact match) for regulated or safety‑critical cases, and more flexible ones (in‑order, any‑order) for exploratory tasks.[1]

## Evaluation methods: test files vs evalsets

| Aspect                    | Test file approach                                   | Evalset approach                                                |
|---------------------------|------------------------------------------------------|------------------------------------------------------------------|
| Granularity               | Single, simple session per file (unit tests).[1] | Multiple sessions per file, can be long and complex (integration tests).[1] |
| Session complexity        | Simple, focused interactions.[1]              | Complex, multi‑turn conversations.[1]                      |
| Frequency                 | Run often during active development.[1]       | Run less frequently due to cost/time.[1]                   |
| File structure            | One session with turns, expected tool use, reference response; *.test.json suffix.[1] | Multiple “evals”, each with turns, expected tool use, intermediate responses, reference, initial session state.[1] |

- Both methods store expected tool use and reference responses; evalsets add named evals and initial session state for richer scenarios.[1]

## Creating and managing eval data

- Test file: define a single session (query turns, expected_tool_use, reference response), save as something like evaluation.test.json (only *.test.json is required).[1]
- Evalset: define multiple evals, each with name, initial state, turns, expected tool use, intermediate responses, and final reference; useful for capturing realistic flows.[1]
- Web UI tools let you capture current or past saved sessions and convert them into evals inside an evalset.[1]

## Ways to run evaluations

- Web UI: interactively run evaluations from the Eval tab, add sessions to evalsets, and compare versions of the agent.[1]
- Programmatic (pytest): use test files in pytest to integrate evaluations into CI/CD and larger test suites.[1]
- Command Line Interface: run evaluations on an existing evalset file directly from the CLI.[1]

## Advanced options

- Initial session state: load session details from a file and pass to AgentEvaluator.evaluate to control starting context.[1]
- Tracing: enable Cloud Trace integration by setting `AF_TRACE_TO_CLOUD=1` in your .env file to debug and observe evaluation runs.[1]

[1](https://www.skills.google/paths/3273/course_templates/1275/video/606601)




      
    







