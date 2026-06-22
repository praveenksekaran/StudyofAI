https://docs.langchain.com/oss/python/langchain/agents

# LangChain
There are two different frameworks for creating agents: LangChain agents and deep agents. Both LangChain and deep agents provide you with fine-grained control over tools, memory, and more. The main difference between both is that deep agents come with a range of commonly useful capabilities already built in, such as planning, file system tools, and subagents.
Use deep agents when you want maximum capability with minimal setup; choose LangChain agents when you need fine-grained control.

#### Agent 
Agent takes harness (Model, tools, System_prompt, checkpointer=InMemorySaver, config=threadid, context=userid)

agent can be invoked agent.invoke() or intermediate steps can be streamed agent.stream() or event streams agent.stream_events()

#### Middleware
Middleware is the primitive for customization: each piece handles one concern, hooks into the agent loop at the right moment, and composes freely with any other
As agents take on complex work, they need support across a few key areas. The middleware ecosystem covers each:
Execution environment, Context management, Planning and delegation, Fault tolerance, Guardrails, Steering
```
from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search],
  //Execution environment
    middleware=[FilesystemMiddleware(backend=StateBackend())],
  //Context management
    MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
    SkillsMiddleware(backend=backend, sources=["./skills/"]),
  //Planning and delegation  
    TodoListMiddleware(),
    SubAgentMiddleware(backend=StateBackend(), subagents=[researcher])
)
```
#### Memory 
**Short Term Memory**
To add short-term memory (thread-level persistence) to an agent, you need to specify a `checkpointer` when creating an agent. 
In production, use a checkpointer backed by a database (postgreSql). 

By default, agents use `AgentState` to manage short term memory, specifically the conversation history via a messages key
You can extend AgentState to add additional fields. Custom state schemas are passed to create_agent using the `state_schema` parameter.

**Solution to Manage State**
- Trim messages : Remove first or Last N messages before calling messages
- Delete Messages: Delete messages from langraph message state
- Summarize messages: summarize and replace history
- custom strategies: Message filtering, etc

#### Structured Output
provide your structued output class or pydentic or json.

#### Guardrails 
Guardrails help you build safe, compliant AI applications by validating and filtering content at key points in your agent’s execution. 
They can detect sensitive information, enforce content policies, validate outputs, and prevent unsafe behaviors before they cause problems.

Common use cases include:
1. Preventing PII leakage
2. Detecting and blocking prompt injection attacks
3. Blocking inappropriate or harmful content
4. Enforcing business rules and compliance requirements
5. Validating output quality and accuracy

Guardrails can be implemented using two complementary approaches:
1. Deterministic guardrails
Use rule-based logic like regex patterns, keyword matching, or explicit checks. Fast, predictable, and cost-effective, but may miss nuanced violations.
2. Model-based guardrails
Use LLMs or classifiers to evaluate content with semantic understanding. Catch subtle issues that rules miss, but are slower and more expensive.

#### Build in
1. PII Detectors
middleware=PIIMiddleware("email",strategy="redact/mask/block",apply_to_input=True,) or
PIIMiddleware("pi_key",detector=r"sk-[a-zA-Z0-9]{32}",strategy="block",apply_to_input=True,]

2. Human-in-the-loop
```
agent = create_agent(
    model="gpt-5.4",
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Require approval for sensitive operations
                "send_email": True,
                "delete_database": True,
                # Auto-approve safe operations
                "search": False,
            }
        ),
    ],
    # Persist the state across interrupts
    checkpointer=InMemorySaver(),
)
```

3. Custom guardrails
use Before agent and After agent hooks to validate request and response

4. Combine multiple guardrails
```
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-5.4",
    tools=[search_tool, send_email_tool],
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),

        # Layer 2: PII protection (before and after model)
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # Layer 3: Human approval for sensitive tools
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),

        # Layer 4: Model-based safety check (after agent)
        SafetyGuardrailMiddleware(),
    ],
)
```

# LangGraph
## Quick start 
Use the `Graph API` if you prefer to define your agent as a graph of nodes and edges. Use the `Functional API` if you prefer to define your agent as a single function.

#### Graph API
[example](https://docs.langchain.com/oss/python/langgraph/quickstart#full-code-example)
```
# 1.Define the tools with @tools and model functions

# 2.Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# 3.Define state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# 4.Define model node
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

# 5.Define tool node
def tool_node(state: dict):
    """Performs the tool call"""

# 6.Define end logic
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

# 7.Build and compile the agent

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])

# Compile the agent
agent = agent_builder.compile()

# Show the agent
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()

```

#### Local server
Langgraph can be run locally for Dev & Testing with InMemory. For Prod Deployment use LangSmith.
use Studio to connect to Langgraph API server to vizualize, interact and debug app locally.

#### Designing the workflow
1. Step: Map out your workflow as discrete steps
Start by identifying the distinct steps in your process. Each step will become a node (a function that does one specific thing). Then, sketch how these steps connect to each other.

2. Identify what each step needs to do 
For each node in your graph, determine what type of operation it represents and what context it needs to work properly.
LLM steps, Data steps, Action or user inputs

3. Design your state
State is the shared memory accessible to all nodes in your agent. Think of it as the notebook your agent uses to keep track of everything it learns and decides as it works through the process.

4. Build your nodes
Now we implement each step as a function. A node in LangGraph is just a Python function that takes the current state and returns updates to it

Handle errors appropriately. 

5. Wire it together
Now we connect our nodes into a working graph. Since our nodes handle their own routing decisions, we only need a few essential edges


#### Human in the loop
https://www.langchain.com/langgraph

interrupt()

## Capabilities
#### Persistant Memory
LangGraph has a built-in persistence layer that saves graph state as `checkpoints`. When you compile a graph with a checkpointer, a snapshot of the graph state is saved at every step of execution, organized into threads. This enables human-in-the-loop workflows, conversational memory, time travel debugging, and fault-tolerant execution.

#### Fault tolerance
When a node fails—from a slow external API, a transient network error, or an unhandled exception—LangGraph gives you three composable mechanisms to respond:
- Retries — automatically re-run failed attempts based on exception type and backoff settings
- Timeouts — cap how long a single attempt may run
- Error handling — run a recovery function after all retries are exhausted

#### Test
https://docs.langchain.com/oss/python/langchain/test

# LangSmith

#### Deployment 


#### Observability
