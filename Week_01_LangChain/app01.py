import os
from langchain.llms import OpenAI
import streamlit as st

st.title('LangChain')
input_text = st.text_input('Input text', 'The quick brown fox jumps over the lazy dog.')
llm = OpenAI(temperature=0.8)

if input_text:
    output_text = llm(input_text)
    st.write(output_text)