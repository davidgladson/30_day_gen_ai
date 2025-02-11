import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st


st.title('Player Search')
input_text = st.text_input('Input text', 'Sachin Tendulkar')

# prompt template
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"
)

# LLM Chain
llm_chain = LLMChain(
    llm=OpenAI(temperature=0.8),
    prompt=first_input_prompt,
    verbose=True)

llm = OpenAI(temperature=0.8)

if input_text:
    output_text = llm_chain.run(input_text)
    st.write(output_text)