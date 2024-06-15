import os

def setupEnv():
    # set api keys here
    os.environ['TAVILY_API_KEY'] = "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"

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = "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"
    return
