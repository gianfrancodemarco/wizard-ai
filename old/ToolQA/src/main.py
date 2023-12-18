import argparse
import datetime
import logging
import os
import sys
from logging import FileHandler

import jsonlines
from agents.react_agent import ReactAgent
from tqdm import tqdm
from llm.llm_client import LLMClientFactory, LLM_MODELS

sys.path.append(os.path.join(os.path.dirname(__file__)))

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

logging_path = os.path.join(
    os.path.dirname(__file__),
    "logs",
    f"{datetime_string}.log"
)
if not os.path.exists(os.path.dirname(logging_path)):
    os.makedirs(os.path.dirname(logging_path))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        FileHandler(logging_path),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


class Executor:

    def __init__(self) -> None:
        self.ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
        self.__parse_args__()
        self.__load_data__()

        llm = LLMClientFactory.create(
            model_name=self.args.model,
            url="https://6e34-34-124-244-176.ngrok.io"
        )
        self.agent = ReactAgent(
            self.args.prompt,
            self.args.path,
            react_llm=llm
        )

    def __parse_args__(self):
        parser = argparse.ArgumentParser("")
        parser.add_argument("--dataset", type=str, default="flights")
        parser.add_argument("--hardness", type=str, default="easy")
        parser.add_argument("--openai_api_key", type=str,
                            default="<OPENAI_API_KEY>")
        parser.add_argument("--path", type=str,
                            default="/home/gianfranco/Desktop/uni/ToolQA")
        parser.add_argument("--wolframalpha_api_key",
                            type=str, default="<WOLFALPHA_API_KEY>")
        parser.add_argument("--debug", type=bool, default=False)
        parser.add_argument("--debug_id", type=int, default=0)
        parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
        parser.add_argument("--prompt", type=str, default="easy")
        self.args = parser.parse_args()
        os.environ["OPENAI_API_KEY"] = self.args.openai_api_key

    def __load_data__(self):
        self.DATASET_PATH = os.path.join(self.ROOT_PATH, "data", "questions",
                                         self.args.hardness, f"{self.args.dataset}-{self.args.hardness}.jsonl")

        with open(self.DATASET_PATH, "r") as f:
            self.data = [item for item in jsonlines.Reader(f)]

    def execute(self):

        summary = {
            "correct": 0,
            "incorrect": 0,
            "halted": 0
        }

        for i, item in enumerate(tqdm(self.data)):
            log.info(f"Question {i+1}/{len(self.data)}")
            answer = self.agent.run(
                item["question"],
                item["answer"],
                item["qid"]
            )
            log.info(f"Answer: {answer.predicted_answer}")
            log.info("---------")

            if self.agent.is_halted(answer):
                summary["halted"] += 1
            elif answer.is_correct():
                summary["correct"] += 1
            else:
                summary["incorrect"] += 1

        log.info(f"Summary: {summary}")

if __name__ == "__main__":
    executor = Executor()
    executor.execute()

#     agent_cls = ReactAgent
#     n = 1
#     log = ''
#     trial = 0
#     unanswered_questions = []
#     agents = []
#     for i in range(len(contents)):
#         agent = agent_cls(args, contents[i]['question'], contents[i]['answer'])
#         try:
#             agent.run()
#             print(f'Answer: {agent.key}')
#             print('---------')
#             log = f"""
# ########################################
# BEGIN TRIAL {contents[i]['qid']}
# #######################################
# """
#             log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
#             with open(os.path.join(logs_dir, contents[i]['qid']+'.txt'), 'w') as f:
#                 f.write(log)
#         except:
#             print('Error when computing answer for {}.'.format(contents[i]['qid']))
#             print('---------')
#             log = f"""
# ########################################
# BEGIN TRIAL {contents[i]['qid']}
# #######################################
# """
#             log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
#             log += 'ERROR!'
#             with open(os.path.join(logs_dir, contents[i]['qid']+'.txt'), 'w') as f:
#                 f.write(log)
#             unanswered_questions.append(contents[i]['qid'])
#         agents.append(agent)
#     trial += 1
#     log += log_react_trial(agents, trial)
#     correct, incorrect, halted = summarize_react_trial(agents)
#     print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
#     print('Unanswered questions: {}'.format(unanswered_questions))
#     # save_agents(agents, os.path.join(root, 'ReAct', 'agents'))
