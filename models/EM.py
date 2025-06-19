# from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
# import streamlit as st

# embedding = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004", google_api_key='AIzaSyCyXr2KjwW58Vm0bewJ_sGEau8C1WS_QNQ')
# from langchain.embeddings import HuggingFaceEmbeddings

# # Load an offline embedding model
# embedding = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5"
# )
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import torch

class VietnameseSBERTEmbeddings(Embeddings):
    def __init__(self, model_name="AITeamVN/Vietnamese_Embedding", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=True,
            device=self.device
        )

    def embed_query(self, text):
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            batch_size=1,
            show_progress_bar=False,
            device=self.device
        )[0]
embedding = VietnameseSBERTEmbeddings()
