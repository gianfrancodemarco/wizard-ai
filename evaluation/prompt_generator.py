from textwrap import dedent

from tools.form_tools import *
import json

configs = {
    "GoogleCalendarCreator": {
        "tool": GoogleCalendarCreator(),
        "task": "create an event on Google Calendar"
    },
    "GoogleCalendarRetriever": {
        "tool": GoogleCalendarRetriever(),
        "task": "retrieve events from Google Calendar"
    },
    "GmailSender": {
        "tool": GmailSender(),
        "task": "send an email"
    },
    "GmailRetriever": {
        "tool": GmailRetriever(),
        "task": "retrieve emails"
    }
}


def create_prompts():

    test_cases = []
    idx = 0

    for key, config in configs.items():

        task, tool = config["task"], config["tool"]

        prompt_header = f"""
                You are impersonating a user testing an AI system.
                Your job is to use the AI system to {task}.
                The data necessary to {task} is the following:
        """

        prompt_footer = f"""
                User:        
        """

        for i in range(3):

            payload = tool.get_random_payload()

            prompt_templates = [
                f"""
                    {prompt_header}
                    {payload}

                    State your will to the AI, without providing the data, and then follow his instructions to complete the job.
                    Act like a very naive user, not giving all of the information at once, and let the AI guide you.

                    {prompt_footer}
                """,
                f"""
                    {prompt_header}
                    {payload}

                    State your will to the AI, without providing the data, and then follow his instructions to complete the job.
                    Act like you don't know what data is necessary to complete the job, and let the AI guide you.
                    Only provide the single information when requested by the AI.
                    Be as direct as possible, and do not provide any unnecessary information.

                    {prompt_footer}
                """,
                f"""
                    {prompt_header}
                    {payload}

                    State your will to the AI, without providing the data, and then follow his instructions to complete the job.
                    Be as direct as possible, and do not provide any unnecessary information.

                    {prompt_footer}
                """,
                f"""
                    {prompt_header}
                    {payload}

                    State your will to the AI, and then follow his instructions to complete the job.
                    Be as direct as possible, and do not provide any unnecessary information.
                    Give all the necessary information to the AI in the first message.
                    {prompt_footer}
                """
            ]

            for j, prompt_template in enumerate(prompt_templates):
                prompt = dedent(prompt_template)

                test_case = {
                    "id": idx,
                    "prompt": prompt,
                    "tool": key,
                    "payload": payload
                }

                test_cases.append(test_case)
                print(test_case)

                idx += 1

    with open(f"evaluation/prompts/prompts.json", "w") as f:
        json.dump(test_cases, f, indent=4, default=str)


create_prompts()
