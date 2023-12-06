import argparse
import json
from tqdm import tqdm
import logging
import requests
from prompts import PROMPTS, ANSWERS
from llm.llm_client import LLMClientFactory, LLMClient, LLMType
from wikidata_client import WikidataClient, WikidataEntityNotFoundException
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log", mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


class Solver():

    def __init__(self) -> None:
        self.LLMClient: LLMClient = LLMClientFactory.create(llm_type=LLMType.GPT3_5_TURBO.value)
        self.retriever = WikidataClient()

    def solve(self, question: str):
        try:
            entities = self.ask_entities(question)
            entities_id = [self.retriever.get_entity_id(
                entity) for entity in entities]
            one_hop_relations = [self.retriever.get_one_hop_relations(
                entity_id) for entity_id in entities_id]
            linearized_relations = self.linearize_relations(
                entities, one_hop_relations)
            answer = self.ask_answer(question, linearized_relations)
            log.info(f"Question: {question}")
            log.info(f"Answer: {answer}")

            answer_entities_ids = []
            if answer == ANSWERS['NO_ANSWER']:
                log.warning("No answer found.")
            else:
                answer_entities = answer.split(', ')
                answer_entities_ids = [self.retriever.get_entity_id(
                    entity) for entity in answer_entities]
                log.info(f"Answer entities: {answer_entities_ids}")

            return answer_entities_ids

        except WikidataEntityNotFoundException as e:
            log.info(e)
            answer = "I don't know."

    def ask_entities(self, question):
        """Ask the LLM to retrieve the entities in the question."""
        question = PROMPTS['ASK_ENTITIES'].replace('<question>', question)
        answer = self.LLMClient.prompt_completion(question)
        entities = answer.split(', ')
        return entities

    def linearize_relations(self, entities, one_hop_relations):
        """Convert the one-hop relations into natural language."""
        linearization = ""
        for entity, one_hop_relation in zip(entities, one_hop_relations):
            linearization += f"{entity}:\n\t"
            for relation_name, relation_values in one_hop_relation.items():
                linearization += f"{relation_name}: {', '.join(relation_values)}\n\t"

        return linearization

    def ask_answer(self, question, linearized_relations):
        """Ask the LLM to answer the question."""
        question = PROMPTS['ASK_ANSWER'].replace('<question>', question).replace(
            '<linearized_relations>', linearized_relations)
        answer = self.LLMClient.prompt_completion(question)
        return answer


class Executor():

    def __init__(self) -> None:
        self.args = self.parse_args()
        self.parse_api_key()
        self.solver = Solver()
        self.data = None

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_path', default=None)
        parser.add_argument('--output_path', default=None)
        parser.add_argument('--chat_log_path', default=None)
        parser.add_argument('--log_path', default=None)
        parser.add_argument('--model_path', default=None)
        parser.add_argument('--debug', action="store_true")
        parser.add_argument('--prompt_path')
        parser.add_argument('--prompt_name', default="chat", )
        parser.add_argument('--bagging_type', default="llm", )
        parser.add_argument('--overwrite', action="store_true")
        parser.add_argument('--device', default=0, help='the gpu device')
        parser.add_argument('--topk', default=10, type=int,
                            help='retrieve the topk score paths')
        parser.add_argument('--max_tokens', default=10,
                            type=int, help='retrieve the topk score paths')
        parser.add_argument(
            '--api_key', default="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", type=str)
        parser.add_argument('--filter_score', default=0.0,
                            type=float, help='the minimal cosine similarity')
        parser.add_argument('--kg_source_path', default=None,
                            help='the sparse triples file')
        parser.add_argument('--ent_type_path', default=None,
                            help='the file of entities type of sparse triples')
        parser.add_argument('--ent2id_path', default=None,
                            help='the sparse ent2id file')
        parser.add_argument('--rel2id_path', default=None,
                            help='the sparse rel2id file')
        parser.add_argument('--ent2name_path', default=None,
                            help='the sparse rel2id file')
        parser.add_argument('--max_triples_per_relation', default=40, type=int)
        parser.add_argument('--max_llm_input_tokens', default=3400, type=int)
        parser.add_argument('--num_process', default=1,
                            type=int, help='the number of multi-process')
        args = parser.parse_args()
        log.info("Start querying the LLM.")
        return args

    def parse_api_key(self):
        api_key = self.args.api_key
        if not api_key.startswith("sk-"):
            api_key_path = self.args.api_key
            with open(api_key_path, "r") as f:
                all_keys = [line.strip('\n') for line in f.readlines()]
                api_key = all_keys[0]
                # all_keys = [line.strip('\n') for line in all_keys]
                # assert len(all_keys) == args.num_process, (len(all_keys), args.num_process)
        os.environ["API_KEY"] = api_key

    def load_data(self, input_path: str):
        with open(input_path, "r") as f:
            self.data = json.load(f)['Questions']
            print(f"Totally {len(self.data)} test examples.")

        # if input_path.endswith('jsonl'):
        #     with open(args.input_path, "r") as f:
        #         all_lines = f.readlines()
        #         all_data = [json.loads(line) for line in all_lines]
        #         print("Totally %d test examples." % len(all_data))

        # elif input_path.endswith('json'):
        #     with open(args.input_path, "r") as f:
        #         all_data = json.load(f)
        #         print("Totally %d test examples." % len(all_data))

    def execute(self):
        self._execute()
        # if args.num_process == 1:
        #     main(args, all_data, idx=-1, api_key=args.api_key)
        # else:
        #     num_each_split = int(len(all_data) / args.num_process)
        #     p = mp.Pool(args.num_process)
        #     for idx in range(args.num_process):
        #         start = idx * num_each_split
        #         if idx == args.num_process - 1:
        #             end = max((idx + 1) * num_each_split, len(all_data))
        #         else:
        #             end = (idx + 1) * num_each_split
        #         split_data = all_data[start:end]
        #         try:
        #             p.apply_async(main, args=(args, split_data, idx, all_keys[idx]))
        #         except Exception as e:
        #             logging.exception(e)

        #     p.close()
        #     p.join()
        #     print("All of the child processes over!")

    def _execute(self):
        self.load_data(self.args.input_path)
        for sample in tqdm(self.data, total=len(self.data)):
            # try:
            question = sample["RawQuestion"]
            self.solver.solve(question)
            # except openai.error.InvalidRequestError as e:
            #     print(e)
            #     continue
            # except Exception as e:
            #     logging.exception(e)
            #     continue

            # chat = sample["ID"] + "\n" + "\n******\n".join(chat_history) + "\nAnswers: " + str(
            #     sample['Answers']) + "\n------------------------------------------\n"
            # fclog.write(chat)

            # count += 1
            # if count < 5:
            #     print(sample['Answers'])
            #     print(prediction)
            #     print("---------------------")
            # sample["Prediction"] = prediction
            # f.write(json.dumps(sample) + "\n")


if __name__ == '__main__':
    executor = Executor()
    executor.execute()
