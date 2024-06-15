from langchain.prompts import PromptTemplate
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


def GetAnswerGrader(localLLM:str = "phi3", temperature:int = 0):
    # LLM
    llm = ChatOllama(model=localLLM, format="json", temperature=temperature)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()
    return answer_grader

