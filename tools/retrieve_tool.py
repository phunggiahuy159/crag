from langchain_community.vectorstores import FAISS
from models.EM import embedding


def retriever():
    vectorstore = FAISS.load_local(
        folder_path="wiki_600", embeddings=embedding, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})
