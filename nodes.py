import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from .state import AgentState
from .tools import TOOLS


load_dotenv()


MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.5")

MAX_STEPS = 8


SYSTEM_PROMPT = """
You are a helpful multi-step AI agent.

You have access to these tools:

1. get_weather
   - Gets current weather information for a city.

2. calculator
   - Performs mathematical calculations.

Your job is to solve the user's request completely.

Rules:

- Decide yourself which tool is required.
- You may call multiple tools.
- If a task requires multiple steps, perform those steps one by one.
- Use results from previous tools when deciding what to do next.
- Do not calculate mathematical results yourself when the calculator tool
  should be used.
- Do not invent weather information.
- If a tool returns an ERROR, handle the error gracefully.
- If a tool fails, explain the limitation and continue if possible.
- When all required information has been collected, provide a clear final answer.
- Do not call tools unnecessarily.
- Keep the final response concise but useful.
"""


model = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
)


model_with_tools = model.bind_tools(TOOLS)


def agent_node(state: AgentState):
    """
    Main reasoning node.

    The LLM looks at the conversation and previous tool results,
    then decides whether another tool is required.
    """

    step_count = state.get("step_count", 0)

    messages = state.get("messages", [])

    system_message = SystemMessage(content=SYSTEM_PROMPT)

    response = model_with_tools.invoke(
        [system_message] + messages
    )

    return {
        "messages": [response],
        "step_count": step_count + 1,
    }


def update_state_node(state: AgentState):
    """
    Collect tool results and store them in custom graph state.
    """

    messages = state.get("messages", [])

    tool_results = list(state.get("tool_results", []))

    for message in messages:

        if message.__class__.__name__ == "ToolMessage":

            content = str(message.content)

            tool_name = getattr(message, "name", "unknown_tool")

            result = f"{tool_name}: {content}"

            if result not in tool_results:
                tool_results.append(result)

    return {
        "tool_results": tool_results
    }


def final_answer_node(state: AgentState):
    """
    Generates a final response after the workflow has enough information.
    """

    messages = state.get("messages", [])

    final_prompt = """
You are now producing the final answer.

Use the complete conversation and all tool results.

Do not call any tools.

Give the user a clear and direct answer.

If any tool failed, explain the failure briefly instead of inventing
information.
"""

    response = model.invoke(
        messages + [
            SystemMessage(content=final_prompt)
        ]
    )

    return {
        "messages": [response],
        "final_answer": response.content,
    }