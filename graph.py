from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langgraph.checkpoint.memory import InMemorySaver

from .state import AgentState
from .tools import TOOLS
from .nodes import (
    agent_node,
    update_state_node,
    final_answer_node,
    MAX_STEPS,
)


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(
    TOOLS,
    handle_tool_errors=True,
)


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_agent(state: AgentState):
    """
    Decide whether the workflow should:

    1. Execute tools
    2. Produce final answer
    3. Stop because maximum steps were reached
    """

    messages = state.get("messages", [])

    if not messages:
        return "final"

    last_message = messages[-1]

    # Prevent infinite loops
    if state.get("step_count", 0) >= MAX_STEPS:
        return "final"

    # If the LLM requested tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise the LLM has enough information
    return "final"


# ============================================================
# BUILD GRAPH
# ============================================================

workflow = StateGraph(AgentState)


# Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("update_state", update_state_node)
workflow.add_node("final", final_answer_node)


# Start
workflow.add_edge(START, "agent")


# Agent decides what happens next
workflow.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "final": "final",
    },
)


# Tool execution
workflow.add_edge("tools", "update_state")


# After tool results are saved,
# send everything back to the agent.
workflow.add_edge("update_state", "agent")


# Final answer
workflow.add_edge("final", END)


# ============================================================
# MEMORY / CHECKPOINTING
# ============================================================

checkpointer = InMemorySaver()


# Compile graph
app = workflow.compile(
    checkpointer=checkpointer
)