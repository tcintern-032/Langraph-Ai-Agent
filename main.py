from graph import app


def print_graph_path():
    """
    Print the graph structure.
    """

    print("\n" + "=" * 60)
    print("LANGGRAPH WORKFLOW")
    print("=" * 60)

    print(
        """
START
  |
  v
AGENT
  |
  +----> TOOLS
  |        |
  |        v
  |   UPDATE STATE
  |        |
  |        v
  |      AGENT
  |
  +----> FINAL
           |
           v
          END
"""
    )

    print("=" * 60)


def print_tool_results(result):
    """
    Display tool results used during the workflow.
    """

    tool_results = result.get("tool_results", [])

    if not tool_results:
        return

    print("\nTool Results:")
    print("-" * 60)

    for item in tool_results:
        print(item)


def run_agent(user_input: str, thread_id: str = "user-1"):
    """
    Run the LangGraph agent.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        },
        config=config,
    )

    print_tool_results(result)

    final_answer = result.get("final_answer")

    if final_answer:
        print("\nAI Agent:")
        print("-" * 60)
        print(final_answer)

    print(
        f"\nWorkflow steps: {result.get('step_count', 0)}"
    )

    return result


def main():
    print("=" * 60)
    print("MULTI-STEP AI AGENT WITH LANGGRAPH")
    print("=" * 60)

    print_graph_path()

    print(
        """
Examples:

1. What's the weather in Lahore?

2. Calculate 25 * 40.

3. What is the weather in Lahore and what is 25% of 1200?

4. Tell me the weather in Islamabad and calculate
   the difference between 1000 and 275.

Type 'exit' to quit.
"""
    )

    thread_id = "user-1"

    while True:

        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in {
                "exit",
                "quit",
                "q"
            }:
                print("\nGoodbye!")
                break

            run_agent(
                user_input=user_input,
                thread_id=thread_id,
            )

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

        except Exception as exc:
            print(
                f"\nApplication error: {exc}"
            )


if __name__ == "__main__":
    main()