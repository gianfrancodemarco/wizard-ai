📦wizard_ai    
 ┣ 📂tests    
 ┃ ┣ 📂behavioural    
 ┃ ┃ ┣ 📜base.py    
 ┃ ┃ ┗ 📜chain.py    
 ┃ ┗ 📂unit    
 ┃ ┃ ┣ 📂conversational_engine    
 ┃ ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┃ ┣ 📜mocks.py    
 ┃ ┃ ┃ ┣ 📜test_helpers.py    
 ┃ ┃ ┃ ┣ 📜test_intent_helpers.py    
 ┃ ┃ ┃ ┗ 📜test_wizard_ai_graph.py    
 ┃ ┃ ┗ 📜conftest.py    
 ┣ 📂wizard_ai    
 ┃ ┣ 📂clients    
 ┃ ┃ ┣ 📂rabbitmq    
 ┃ ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┃ ┣ 📜constants.py    
 ┃ ┃ ┃ ┣ 📜rabbitmq_consumer.py    
 ┃ ┃ ┃ ┗ 📜rabbitmq_producer.py    
 ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┣ 📜google.py    
 ┃ ┃ ┣ 📜google_search.py    
 ┃ ┃ ┗ 📜redis.py    
 ┃ ┣ 📂constants    
 ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┣ 📜message_queues.py    
 ┃ ┃ ┣ 📜message_type.py    
 ┃ ┃ ┗ 📜redis_keys.py    
 ┃ ┣ 📂controllers    
 ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┣ 📜conversations.py    
 ┃ ┃ ┣ 📜google_actions.py    
 ┃ ┃ ┗ 📜google_login.py    
 ┃ ┣ 📂conversational_engine    
 ┃ ┃ ┣ 📂langchain_extention    
 ┃ ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┃ ┣ 📜form_tool.py    
 ┃ ┃ ┃ ┣ 📜helpers.py    
 ┃ ┃ ┃ ┣ 📜intent_helpers.py    
 ┃ ┃ ┃ ┣ 📜tool_executor_with_state.py    
 ┃ ┃ ┃ ┗ 📜wizard_ai_graph.py    
 ┃ ┃ ┣ 📂memory    
 ┃ ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┃ ┗ 📜memory.py    
 ┃ ┃ ┣ 📂tools    
 ┃ ┃ ┃ ┣ 📂google    
 ┃ ┃ ┃ ┃ ┣ 📂calendar    
 ┃ ┃ ┃ ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┃ ┃ ┃ ┣ 📜creator.py    
 ┃ ┃ ┃ ┃ ┃ ┗ 📜retriever.py    
 ┃ ┃ ┃ ┃ ┣ 📂gmail    
 ┃ ┃ ┃ ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┃ ┃ ┃ ┗ 📜retriever.py    
 ┃ ┃ ┃ ┃ ┣ 📂search    
 ┃ ┃ ┃ ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┃ ┃ ┃ ┗ 📜search.py    
 ┃ ┃ ┃ ┃ ┗ 📜__init__.py    
 ┃ ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┃ ┗ 📜python_code_interpreter.py    
 ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┗ 📜engine.py    
 ┃ ┣ 📂models    
 ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┗ 📜chat_payload.py    
 ┃ ┣ 📜__init__.py    
 ┃ ┣ 📜html_processor.py    
 ┃ ┗ 📜main.py    
 ┣ 📜poetry.lock    
 ┗ 📜pyproject.toml