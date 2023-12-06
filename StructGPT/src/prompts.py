ANSWERS = {
    "NO_ANSWER": "<no_answer>"
}
PROMPTS = {
    "ASK_ENTITIES": """
Identify specific instances, referred to as ENTITIES, in the given question. ENTITIES are concrete examples such as "Donald Trump," "Italy," or "Potassium." Exclude generic terms like "country," "team," or "university" from consideration. Provide a response containing only the identified entities, separated by commas. Avoid including any additional information in the output.






        ### Examples

        Question:
        What is the nationality of Joe Biden?

        Entities: Joe Biden

        Question: 
        Who is the son of Lily Evans and James Potter?

        Entities: Lily Evans, James Potter

        ### Actual question

        Question:
        <question>

        Entities: 

    """,
    "ASK_ANSWER": """
        You are a virtual assistance.
        You answer questions ONLY using the data you are given below.
        Your response must only consist of the direct responses to the answer, separated by a comma.
        If you cannot find an answer, return <no_answer>.

        ### Examples
        
        ## Data 
        Italy:
            Capital: Rome
            
        Question: What is the capital of Italy?
        Answer: Rome

        ## Data 
        FAANG:
            members: Facebook, Apple, Amazon, Netflix, Google

        Question: Which companies form the FAANG group?
        Answer: Facebook, Apple, Amazon, Netflix, Google

        
        ### Actual question
        
        ### Data
        <linearized_relations>

        Question: <question>
        Answer:
    """
}