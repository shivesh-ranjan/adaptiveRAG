from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings

def getDocs(docList:list, docType: str) -> list:
    if docType=="urls":
        docs = [WebBaseLoader(url).load() for url in docList]
    else:
        return []
    return docs


def getRetriever(docList:list, docType: str, collectionName:str, chunkSize:int = 500, chunkOverlap:int=0):
    # Load
    docs = getDocs(docList, docType)
    docs_list = [item for sublist in docs for item in sublist]

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunkSize, chunk_overlap=chunkOverlap
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collectionName,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
    )
    retriever = vectorstore.as_retriever()
    return retriever
