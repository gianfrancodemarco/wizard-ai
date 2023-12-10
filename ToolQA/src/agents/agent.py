import re
from logging import getLogger

from agents.utils import *

from prompts import REFLECTION_AFTER_LAST_TRIAL_HEADER
from tools.code import sql_interpreter
from tools.math import calculator
from tools.text import agenda_retriever, scirex_retriever

log = getLogger(__name__)


class Agent:

    def reflect(self,
                strategy: ReflexionStrategy) -> None:
        log.info('Reflecting...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(
                self.question, self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(
                self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += format_reflections(
                self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(
                f'Unknown reflection strategy: {strategy}')
        log.info(self.reflections_str)

    def act(
        self,
        action_context: ActionContext
    ) -> ActionOutput:
        actions = {
            'Finish': lambda argument: None,
            'Calculate': lambda argument: str(calculator.WolframAlphaCalculator(argument)).strip('\n').strip(),
            'RetrieveAgenda': lambda argument: agenda_retriever.query_llm([0], argument).strip('\n').strip(),
            'RetrieveScirex': lambda argument: scirex_retriever.query_llm([0], argument).strip('\n').strip(),
            'LoadDB': lambda argument: self.table_toolkits.db_loader(argument),
            'FilterDB': lambda argument: self.table_toolkits.data_filter(argument),
            'GetValue': lambda argument: self.table_toolkits.get_value(argument),
            'LoadGraph': lambda argument: self.graph_toolkits.load_graph(argument),
            'NeighbourCheck': lambda argument: self.graph_toolkits.check_neighbours(argument),
            'NodeCheck': lambda argument: self.graph_toolkits.check_nodes(argument),
            'EdgeCheck': lambda argument: self.graph_toolkits.check_edges(argument),
            'SQLInterpreter': lambda argument: sql_interpreter.execute(argument),
            'PythonInterpreter': lambda argument: exec(argument)
        }

        exceptions = {
            'invalid_action': 'Invalid Action. Valid Actions are Calculate [<Formula>] RetrieveAgenda[<Content>] RetrieveScirex[<Content>] LoadDB[<DBName>] FilterDB[<Condition>, <Condition>, ...] GetValue[<Column>] LoadGraph[<GraphName>] NeighbourCheck[<GraphName>, <Node>] NodeCheck[<GraphName>, <Node>] EdgeCheck[<GraphName>, <Node1>, <Node2>] SQLInterpreter[<SQLCommand>] PythonInterpreter[<PythonCode>] and Finish[<answer>].',
            'RetrieveAgenda': 'There is no information that can be matched in the database. Please try another query.',
            'RetrieveScirex': 'There is no information that can be matched in the database. Please try another query.',
            'LoadDB': 'The database you want to query in not in the list. Please change another database for query.',
            'FilterDB': 'There is something wrong with the arguments you send for filtering. Please modify it.',
            'GetValue': 'The value you are querying does not exist. Please modify it.',
            'LoadGraph': 'The graph you want to query in not in the list. Please change another graph for query.',
            'NeighbourCheck': 'There is something wrong with the arguments you send for neighbour checking. Please modify it.',
            'NodeCheck': 'There is something wrong with the arguments you send for node checking. Please modify it.',
            'EdgeCheck': 'There is something wrong with the arguments you send for edge checking. Please modify it.',
            'SQLInterpreter': 'There is something wrong with the SQL command you send. Please modify it.',
            'PythonInterpreter': 'There is something wrong with the Python code you send. Please modify it.'
        }

        output = ActionOutput()

        # Retrieve the action
        try:
            action = actions.get(action_context.action_type)
        except:
            output.message = exceptions.get('invalid_action')
            output.success = False
            return

        # Act
        try:
            output.message = action(action_context.argument)
            output.success = True
        except openai.error.RateLimitError:
            output.message = f'OpenAI API Rate Limit Exceeded. Please try again.'
            output.success = False
        except:
            output.message = exceptions.get(action_context.action_type)
            output.success = False

        # if action_context.action_type == 'Finish':
        #     self.step_n += 1
        return output

    def parse_action(self, string):
        pattern = r'^(\w+)\[(.+)\]$'
        match = re.match(pattern, string)
        action_context = None

        if match:
            action_type = match.group(1)
            argument = match.group(2)
            action_context = ActionContext(action_type, argument)

        return action_context
