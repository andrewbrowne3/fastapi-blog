import os

from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db_dir_questions", "chroma_db_with_metadata"
)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)


query = input("context")

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.3},
)
relevant_docs = retriever.invoke(query)


print("\n--relevant documents--")
for i, doc in enumerate(relevant_docs, 1):
    if doc.metadata:
        print(f"Source: {doc.metadata['source']}")
