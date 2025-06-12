from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import streamlit as st

embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key='AIzaSyCyXr2KjwW58Vm0bewJ_sGEau8C1WS_QNQ'
