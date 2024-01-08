from datetime import datetime
from textwrap import dedent
from typing import Tuple


def get_prompt_components(
    tool_name: str,
    information_to_collect: str,
    information_collected: str,
) -> Tuple[str, str, str]:
    prefix = dedent(f"""
        Today is: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        You are a personal assistant. The user is trying to fill data for {tool_name} and you need to help him.

        Kindly ask the user to provide the next missing information using the Final Answer tool.

        You have access to the following tools:
    """)

    FORMAT_INSTRUCTIONS = dedent("""
        Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

        Valid "action" values: "Final Answer" or {tool_names}

        Provide only ONE action per $JSON_BLOB, as shown:

        ```
        {{{{
        "action": $TOOL_NAME,
        "action_input": $INPUT
        }}}}
        ```

        Follow this format:

        Question: human input to the assistant

        (repeat the following Thought/Action/Observation N times)
        Thought: consider previous and subsequent steps
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: action result
        Thought: I know what to respond
        Action:
        ```
        {{{{
        "action": "Final Answer",
        "action_input": "Final response to human"
        }}}}                                     
        ```
    """)

    suffix = dedent(f"""
        Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
        Thought:

        The information you need to collect is the following:
        {information_to_collect}

        The information you have collected so far is:
        {information_collected}

        When you have all the NEEDED information, call {tool_name} with the input data.
    """)

    return prefix, suffix, FORMAT_INSTRUCTIONS