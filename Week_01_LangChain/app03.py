import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import streamlit as st


st.title('Player Search')
input_text = st.text_input('Input text', 'Sachin Tendulkar')

# prompt template
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"
)

# LLM Chain
llm_chain_01 = LLMChain(
    llm=OpenAI(temperature=0.8),
    prompt=first_input_prompt,
    output_key='person',
    verbose=True)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born ?"
)

llm_chain_02 = LLMChain(
    llm=OpenAI(temperature=0.8),
    prompt=second_input_prompt,
    output_key='dob',
    verbose=True)


parent_chain = SequentialChain(chains=[llm_chain_01, llm_chain_02], 
                               input_variables=['name'],
                               output_variables=['person', 'dob'],
                               verbose=True)

if input_text:
    response = parent_chain({"name": input_text})
    st.write(response)