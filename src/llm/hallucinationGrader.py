
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

def GetHallucinationGrader(localLLM: str = "phi3", temperature: int = 0):
    llm = ChatOllama(model=localLLM, format="json", temperature= temperature)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader
