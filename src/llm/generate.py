from langchain import hub
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from src.env.env import LOCAL_LLM



def GetRAGChain():
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOllama(model=LOCAL_LLM)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain
