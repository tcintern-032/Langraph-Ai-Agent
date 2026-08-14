from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """
    Shared state used by all LangGraph nodes.
    """

    messages: Annotated[list[BaseMessage], add_messages]

    # Stores results returned by tools
    tool_results: list[str]

    # Number of agent/tool cycles
    step_count: int

    # Final response
    final_answer: str