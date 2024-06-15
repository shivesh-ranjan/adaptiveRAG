from langchain import hub
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser



def GetRAGChain(localLLM: str = "phi3", temperature: int = 0):
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOllama(model=localLLM, temperature=temperature)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain
