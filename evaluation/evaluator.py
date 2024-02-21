import json
import os
from datetime import datetime
from typing import Any, Dict

from langchain.agents.output_parsers.openai_tools import OpenAIToolAgentAction
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.structured_tools import *

from wizard_ai.conversational_engine.form_agent.form_agent_executor import \
    FormAgentExecutor

TEST_CASES_PATH = os.path.join(
    os.path.dirname(__file__), "prompts/prompts.json")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo-0125")

test_cases = json.loads(open(TEST_CASES_PATH).read())


class MaxIterationsReached(Exception):
    pass


class SuccessfulExecution(Exception):
    pass

def normalize_json(json_data):
    """
    The LLM can call the correct tool with the correct output, but it may differ from the expected one by small details.
    For examples, the dates may be in different formats, or the whitespace may be different, there may be dots at the end of the sentences, etc.

    This function normalizes the JSON string so that it can be compared with the expected output.
    We assume that these small differences are not relevant for the evaluation.
    """

    # Normalize date format
    for key, value in json_data.items():
        if key in ['start', 'end']:
            json_data[key] = datetime.fromisoformat(value.replace('T', ' ')).strftime('%Y-%m-%d %H:%M:%S')
        json_data[key] = json_data[key].replace("\n", " ").replace(".", " ")

    # Normalize whitespace
    json_data = {key: value.strip() if isinstance(value, str) else value for key, value in json_data.items()}
    print(json_data)
    return json_data

class UserLLMForEvaluation:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            verbose=True
        )
        self.history = []

    def execute_first(self, message):
        response = self.llm([SystemMessage(content=message)])
        self.history.extend([
            SystemMessage(content=message),
            response
        ])
        return response.content

    def execute(self, message):
        response = self.llm([
            *self.history,
            HumanMessage(content=message)
        ])
        self.history.append(response)
        return response.content


class FormAgentExecutorForEvaluation:
    """
    This class is used to execute the form agent in the conversational engine.

    Args:
        target_tool_call (Dict[str, Any], optional): The target tool call that we want to reach.
        The executor will raise a SuccessfulExecution exception if the target tool call is reached, or 
        a MaxIterationsReached exception if the maximum number of iterations is reached.
    """

    def __init__(
        self,
        target_tool_call: Dict[str, Any] = None,
    ):
        self.target_tool_call = target_tool_call
        self.max_iterations = 10
        self.current_iteration = 1
        self.tools = [
            GoogleCalendarCreator(),
            GoogleCalendarRetriever(),
            GmailRetriever(),
            GmailSender()
        ]
        self.graph = FormAgentExecutor(tools=self.tools)
        self.state = {
            "input": "",
            "chat_history": [],
            "intermediate_steps": [],
            "active_form_tool": None
        }

    def execute(self, input_data):

        inputs = {
            "input": input_data,
            "chat_history": self.state["chat_history"],
            "intermediate_steps": self.state["intermediate_steps"],
            "active_form_tool": self.state["active_form_tool"]
        }

        for output in self.graph.app.stream(inputs, config={"recursion_limit": 25}):
            for key, value in output.items():
                self.check_successful_execution(key, value)
        output = self.graph.parse_output(output)

        # Update state
        self.state.update({
            "chat_history": [
                *self.state["chat_history"],
                HumanMessage(content=input_data),
                AIMessage(content=output)
            ],
            "active_form_tool": value["active_form_tool"]
        })

        if self.current_iteration >= self.max_iterations:
            raise MaxIterationsReached()

        self.current_iteration += 1

        return output

    def check_successful_execution(self, key, value):
        """
        Check if the target tool call is reached. If it is, raise a SuccessfulExecution exception.
        This is done by checking that when the agent is confirming the tool call, the current form of the FormTool is the same as the expected one.
        When checking the form, the JSON is normalized string so that it can be compared with the expected output.
        """


        if key != "agent":
            return

        if not isinstance(value["agent_outcome"], OpenAIToolAgentAction):
            return

        agent_outcome = value["agent_outcome"]
        target_tool_name = f"{self.target_tool_call['tool']}Finalize"

        if not agent_outcome.tool == target_tool_name:
            return

        if not agent_outcome.tool_input == {'confirm': True}:
            return
        
        target_tool = next(filter(
            lambda tool: tool.name == target_tool_name,
            self.graph._tools
        ))

        normalized_expected_output = normalize_json(self.target_tool_call['payload'])
        normalized_actual_output = normalize_json(json.loads(target_tool.form.model_dump_json()))

        if normalized_expected_output == normalized_actual_output:
            raise SuccessfulExecution()


for test_case in test_cases[:1]:
    try:
        # Get user message

        prompt = test_case["prompt"]
        tool = test_case["tool"]
        payload = test_case["payload"]

        user_model = UserLLMForEvaluation()
        system_model = FormAgentExecutorForEvaluation(
            target_tool_call={
                "tool": tool,
                "payload": payload
            }
        )

        user_response = user_model.execute_first(prompt)
        print(user_response)

        while True:
            system_response = system_model.execute(user_response)
            print(system_response)
            user_response = user_model.execute(system_response)
            print(user_response)
    except SuccessfulExecution:
        print("Successful execution")
    except MaxIterationsReached:
        print("Max iterations reached")