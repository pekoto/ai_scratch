# Run with `streamlit run app.py`
# Note: I think this is using old syntax.

import os
from getpass import getpass

import streamlit as st
# Switch to OpenAI model
#from langchain_community.llms import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper

# TODO: Upgrade to latest syntax
# os.environ['OPENAI_API_KEY'] = API_KEY
api_key = getpass()

st.title('🎥YouTube Script Creator')
prompt = st.text_input('Enter Prompt')

# Prompt template
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Give me a single YouTube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write me a YouTube script based on this title: {title}, and using this wikipedia research:{wikipedia_research}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMs
# llm = OpenAI(temperature=0.9)
llm = GoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=api_key)

title_chain = LLMChain(
    llm=llm,
    prompt=title_template,
    verbose=True,
    output_key='title',
    memory=title_memory,
)

script_chain = LLMChain(
    llm=llm,
    prompt=script_template,
    verbose=True,
    output_key='script',
    memory=script_memory,
)

# Set up sequential chain
# sequential_chain = SequentialChain(
#     chains=[title_chain, script_chain],
#     input_variables=['topic'],
#     output_variables=['title', 'script'],
#     verbose=True)

wiki = WikipediaAPIWrapper()

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    # Is using sequential chain
    # response = sequential_chain({'topic': prompt})
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wiki History'):
        st.info(wiki_research)
