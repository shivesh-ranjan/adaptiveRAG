import os

LOCAL_LLM = "mistral"

def setupEnv():
    # set api keys here

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = "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"
    return
