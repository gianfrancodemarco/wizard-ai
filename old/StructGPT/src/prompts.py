ANSWERS = {
    "NO_ANSWER": "<no_answer>"
}
PROMPTS = {
    "ASK_ENTITIES": """
        Find ENTITIES in the question asked to you.
        An ENTITY is a SPECIFIC instance of something, for example "Donald Trump", "Italy", "Potassium".
        Generic words like "country" or "team" or "university" must not be recognized as entities.
        ENTITIES can also be numeric identifiers, like "333312" or "Q333312".
        Your response must ONLY consist of the entities separated by a comma.
        Do not add any other output.

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