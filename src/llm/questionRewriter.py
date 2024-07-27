from langchain.prompts import PromptTemplate
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from src.env.env import LOCAL_LLM


def GetQuestionRewriter():
    # LLM
    llm = ChatOllama(model=LOCAL_LLM, temperature=0)

    # Prompt 
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the initial and formulate an improved question. \n
        Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
        input_variables=["generation", "question"],
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter

