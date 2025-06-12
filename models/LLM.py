from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key='AIzaSyCyXr2KjwW58Vm0bewJ_sGEau8C1WS_QNQ'
)
