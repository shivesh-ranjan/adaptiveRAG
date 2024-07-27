# from src.index.index import getRetriever
# from src.llm.answerGrader import GetAnswerGrader
# from src.llm.questionRewriter import GetQuestionRewriter
# from src.llm.router import GetRouter
# from src.llm.retrievalGrader import GetRetrievalGrader
# from src.llm.generate import GenerateResponse
# from src.llm.hallucinationGrader import GetHallucinationGrader
#
# if __name__=="__main__":
#     question_router = GetRouter()
#     retriever = getRetriever(
#             docList=[
#                 "https://lilianweng.github.io/posts/2023-06-23-agent/",
#                 "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#                 "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
#             ],
#             docType="urls",
#             collectionName="rag-chroma"
#     )
#     question = "llm agent memory"
#     docs = retriever.invoke(question)
#     doc_txt = docs[1].page_content
#     print(question_router.invoke({"question": question}))
#
#     retrieval_grader = GetRetrievalGrader()
#     print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
#
#     generation = GenerateResponse(question, docs)
#     print(generation)
#
#     hallucination_grader = GetHallucinationGrader()
#     print(hallucination_grader.invoke({"documents": docs, "generation": generation}))
#
#     answer_grader = GetAnswerGrader()
#     print(answer_grader.invoke({"question": question,"generation": generation}))
#
#     question_rewriter = GetQuestionRewriter()
#     print(question_rewriter.invoke({"question": question}))

from pprint import pprint
from src.graph.build import GetApp
from src.env.env import setupEnv

# Run
setupEnv()
app = GetApp()
while True:
    question = input(">>> ")
    if question in ["exit", "Exit", "EXIT"]:
        break
    else:
        inputs = {"question": question}
        for output in app.stream(inputs):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
                pprint("\n---\n")
                # Final generation
                #if "generation" in value.keys(): 
        pprint(value["generation"])
