# Langraph-Ai-Agent
# Multi-Step AI Agent with LangGraph
A multi-step AI agent built with **Python, LangGraph, LangChain, and OpenAI**. The agent can decide which tools to use, execute multiple tools, pass results through shared LangGraph state, handle errors, and continue the workflow until it can generate a final response.

## Project Overview

This project demonstrates how to build a structured AI Agent workflow using LangGraph.

The agent supports:

* LLM-based decision making
* Multiple tools
* Multi-step workflows
* LangGraph State
* Conditional routing
* Tool execution using `ToolNode`
* Tool result handling
* Error handling
* Conversation history
* Graph execution visualization
* Optional LangSmith tracing

The main example uses two tools:

1. **Weather Tool** — Gets current weather information for a city.
2. **Calculator Tool** — Performs mathematical calculations.

---

# 📁 Project Structure

```text
multi-step-ai-agent/
│
├── app/
│   ├── __init__.py
│   ├── state.py
│   ├── tools.py
│   ├── nodes.py
│   ├── graph.py
│   └── main.py
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Technologies Used

* Python 3.11
* LangChain
* LangGraph
* OpenAI
* Open-Meteo API
* Python Requests
* python-dotenv
* LangSmith

---

# Topics Covered

This project covers the following AI Engineering concepts:

### 1. LangGraph State

A shared state object stores:

* Conversation messages
* Tool results
* Number of workflow steps
* Final answer

Example:

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_results: list[str]
    step_count: int
    final_answer: str
```

---

### 2. Nodes

The graph contains four main nodes:

```text
Agent
Tools
Update State
Final
```

Each node performs a specific task.

---

### 3. Edges

Edges define how the workflow moves between nodes.

Example:

```text
START → Agent → Tools → Update State → Agent
```

---

### 4. Conditional Routing

The agent decides whether it needs to use a tool or generate the final answer.

```text
              Agent
             /     \
            /       \
         Tools      Final
           |
      Update State
           |
         Agent
```

---

### 5. Tool Nodes

The project uses LangGraph's `ToolNode` to execute tools.

Available tools:

```text
get_weather
calculator
```

---

### 6. Multi-Step Agent Workflow

The agent can use more than one tool during a single request.

Example:

```text
User Request
     ↓
Agent
     ↓
Weather Tool
     ↓
Update State
     ↓
Agent
     ↓
Calculator Tool
     ↓
Update State
     ↓
Agent
     ↓
Final Answer
```

---

# Available Tools

## Weather Tool

The weather tool uses the Open-Meteo API.

Example request:

```text
What's the weather in Lahore?
```

The tool can return information such as:

* Temperature
* Feels-like temperature
* Humidity
* Wind speed
* Weather condition

No separate weather API key is required.

---

## Calculator Tool

The calculator tool performs mathematical operations.

Examples:

```text
25 * 40
```

```text
1200 * 0.25
```

```text
(100 + 50) / 2
```

The calculator also handles errors such as division by zero.
# Running the Project

After activating the virtual environment:

```bash
python -m app.main
```

You should see:

```text
============================================================
MULTI-STEP AI AGENT WITH LANGGRAPH
============================================================
```

The application will then wait for your input.

---

# Example Queries

## Example 1 — Weather

```text
What's the weather in Lahore?
```

Workflow:

```text
User
 ↓
Agent
 ↓
Weather Tool
 ↓
Update State
 ↓
Agent
 ↓
Final Answer
```

---

## Example 2 — Calculator

```text
Calculate 25 * 40
```

Expected result:

```text
1000
```

Workflow:

```text
User
 ↓
Agent
 ↓
Calculator Tool
 ↓
Update State
 ↓
Agent
 ↓
Final Answer
```

---

# Multi-Step Example

This is the main example for this project.

Ask:

```text
What is the weather in Lahore and what is 25% of 1200?
```

The agent can perform multiple steps:

```text
User Request
      ↓
    Agent
      ↓
Weather Tool
      ↓
Weather Result
      ↓
Update State
      ↓
    Agent
      ↓
Calculator Tool
      ↓
Calculation Result
      ↓
Update State
      ↓
    Agent
      ↓
 Final Answer
```

The calculation result is:

```text
25% of 1200 = 300
```

The weather information is retrieved at runtime.

---

# ❌ Error Handling

The application handles tool errors gracefully.

For example:

```text
Calculate 100 / 0
```

The calculator returns:

```text
ERROR: Cannot divide by zero.
```

Another example:

```text
What's the weather in xyzabc123?
```

The weather tool can return an error when the city cannot be found.

The agent should explain the problem instead of crashing the application or inventing an answer.

---

# LangGraph State

The project uses a shared state:

```python
class AgentState(TypedDict, total=False):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    tool_results: list[str]

    step_count: int

    final_answer: str
```

The state allows information to move between different nodes.

For example:

```text
Agent
  ↓
Tool
  ↓
Tool Result
  ↓
State
  ↓
Agent
```

This is what allows the agent to continue reasoning after a tool has been executed.

---

# Conditional Routing

The graph checks whether the LLM requested a tool.

If a tool call exists:

```text
Agent → Tools
```

If no tool call exists:

```text
Agent → Final
```

Conceptually:

```text
             ┌─────────────┐
             │    Agent    │
             └──────┬──────┘
                    │
              Conditional
               Routing
               /       \
              /         \
             ↓           ↓
         Tools          Final
           ↓
     Update State
           ↓
         Agent
```

This creates the multi-step loop.

---

# 🔄 Conversation History

The project uses LangGraph checkpointing:

```python
checkpointer = InMemorySaver()
```

A thread ID is used:

```python
config = {
    "configurable": {
        "thread_id": "user-1"
    }
}
```

This allows the agent to maintain conversation history within the same thread.

Example:

```text
You:
What's the weather in Lahore?

AI:
The current weather is ...

You:
What about Karachi?

AI:
The current weather in Karachi is ...
```

---

# Graph Visualization

The project can display the LangGraph workflow.

Conceptually:

```text
START
  |
  v
AGENT
  |
  +-----------> TOOLS
  |                |
  |                v
  |          UPDATE STATE
  |                |
  |                v
  +------------- AGENT
                   |
                   v
                 FINAL
                   |
                   v
                  END
```

The graph can also be exported as Mermaid syntax using:

```python
app.get_graph().draw_mermaid()
```

---

# LangSmith Tracing

LangSmith can be enabled for tracing and debugging the agent.

Update `.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=multi-step-ai-agent
```

Then run:

```bash
python -m app.main
```

LangSmith can help inspect:

* LLM calls
* Tool calls
* Tool results
* Workflow execution
* Agent steps
* Errors
* Execution time

A multi-step trace may look conceptually like:

```text
Agent
  |
  ├── Weather Tool
  |
  ├── Update State
  |
  ├── Agent
  |
  ├── Calculator Tool
  |
  ├── Update State
  |
  ├── Agent
  |
  └── Final
```

---

# 🧩 File Responsibilities

## `state.py`

Defines the shared LangGraph state.

```text
AgentState
```

---

## `tools.py`

Contains the tools:

```text
get_weather()
calculator()
```

---

## `nodes.py`

Contains the main agent logic:

```text
agent_node()
update_state_node()
final_answer_node()
```

---

## `graph.py`

Builds and compiles the LangGraph workflow.

Contains:

```text
StateGraph
ToolNode
Conditional Routing
Edges
Checkpointer
```

---

## `main.py`

Runs the application and provides the command-line interface.

---

# 🏗️ Architecture

```text
                    ┌──────────────────┐
                    │   User Request   │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │    Agent Node    │
                    │      OpenAI      │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Conditional      │
                    │ Routing          │
                    └───────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ↓                     ↓
        ┌────────────────┐     ┌───────────────┐
        │   Tool Node    │     │  Final Node   │
        └───────┬────────┘     └───────┬───────┘
                ↓                      ↓
        ┌────────────────┐            END
        │ Update State   │
        └───────┬────────┘
                ↓
        ┌────────────────┐
        │   Agent Node   │
        └───────┬────────┘
                │
                └──────────→ Continue
```

---

# Learning Objectives

After completing this project, you should understand:

* What LangGraph State is
* How to create a `StateGraph`
* How nodes work
* How edges connect nodes
* How conditional routing works
* How `ToolNode` executes tools
* How an LLM selects tools
* How tool results are passed back to the agent
* How multi-step agent workflows work
* How to handle tool errors
* How conversation history works
* How to visualize a graph
* How LangSmith can trace an agent

---

# Challenge Requirements

The project satisfies the task requirements:

| Requirement                 | Status |
| --------------------------- | ------ |
| LangGraph State             | ✅      |
| Nodes & Edges               | ✅      |
| Tool Nodes                  | ✅      |
| Conditional Routing         | ✅      |
| Multi-Step Workflow         | ✅      |
| At least 2 Tools            | ✅      |
| Tool Selection by Agent     | ✅      |
| Pass Results Through State  | ✅      |
| Continue Until Final Answer | ✅      |
| Error Handling              | ✅      |
| Conversation History        | ✅      |
| Graph Execution Path        | ✅      |
| LangSmith Tracing           | ✅      |
| Graph Visualization         | ✅      |

---

# Future Improvements

Possible extensions include:

* Add a web search tool
* Add a calculator with more operations
* Add a database tool
* Add a RAG retrieval tool
* Add PDF document search
* Add persistent database checkpointing
* Build a FastAPI backend
* Build a Streamlit frontend
* Add authentication
* Add human-in-the-loop approval
* Add more advanced conditional routing
* Add multiple specialized agents
* Deploy the application to the cloud

---

# Example Final Workflow

For the request:

```text
What is the weather in Lahore and what is 25% of 1200?
```

The complete agent workflow is:

```text
                    USER
                     |
                     ↓
                   AGENT
                     |
                     ↓
              ┌──────────────┐
              │ Need Weather │
              └──────┬───────┘
                     ↓
              WEATHER TOOL
                     |
                     ↓
              WEATHER RESULT
                     |
                     ↓
               UPDATE STATE
                     |
                     ↓
                   AGENT
                     |
                     ↓
             ┌────────────────┐
             │ Need Calculator│
             └───────┬────────┘
                     ↓
               CALCULATOR
                     |
                     ↓
             CALCULATION RESULT
                     |
                     ↓
               UPDATE STATE
                     |
                     ↓
                   AGENT
                     |
                     ↓
               FINAL ANSWER
                     |
                     ↓
                    END
```
