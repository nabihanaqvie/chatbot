import streamlit as st
import boto3
import requests
import json
import os
from typing import List
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pinecone
from datasets import load_dataset
import pandas as pd


if 'generated' not in st.session_state: 
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []





endpoint_name = "huggingface-pytorch-tgi-inference-2023-11-21-00-38-12-570" 
st.sidebar.image("berkeley1.png", width=300)
st.sidebar.markdown("""<div style="text-align:center;">""", unsafe_allow_html=True) 
st.sidebar.title("W210 - Capstone")
st.sidebar.markdown("""[Sarah Hoover](https://www.linkedin.com/in/sarah-hoover-08816bba/), [Nabiha Naqvie](https://www.linkedin.com/in/nabiha-naqvie-22765612a/), [Bindu Thota](https://www.linkedin.com/in/bindu-thota/), and [Dave Zack](https://www.linkedin.com/in/dave-zack/)""")
st.sidebar.markdown("""</div>""", unsafe_allow_html=True)

st.sidebar.markdown("""---""")
st.sidebar.header("About")
st.sidebar.info("""
    Insert Info

""")

st.sidebar.markdown("""---""")

st.sidebar.header("Source Code")
st.sidebar.markdown("""
    View the source code on [GitHub](https://github.com/nabihanaqvie/chatbot/blob/main/app_w210.py)  
""", unsafe_allow_html=True)
st.sidebar.markdown("""---""")

# Import the dataset to get the character names/bios
npc_train = load_dataset("amaydle/npc-dialogue", split="train")
npc_test = load_dataset("amaydle/npc-dialogue", split="test")

# Automatically splits it into train and test for you - let's ignore that for now and just combine them as one

# First, transform them into pandas DFs
train = pd.DataFrame(data = {'name': npc_train['Name'], 'bio':npc_train['Biography'], 'query':npc_train['Query'], 'response':npc_train['Response'], 'emotion':npc_train['Emotion']})
test = pd.DataFrame(data = {'name': npc_test['Name'], 'bio':npc_test['Biography'], 'query':npc_test['Query'], 'response':npc_test['Response'], 'emotion':npc_test['Emotion']})

# Now combine into a single df
npc = pd.concat([train, test])

# Create a character-level dataset, since the characters show up multiple times in the dataset
character_level_dataset = npc[['name', 'bio']]
character_level_dataset.drop_duplicates(inplace=True)

# Get a list of just the names
character_names = list(pd.unique(character_level_dataset['name']))

# Set up the sentence encoder for RAG
rag_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to encode the input for retrieval
def embed_docs(docs: List[str]) -> List[List[float]]:
    out = rag_encoder.encode(docs)
    return out.tolist()

# Connect to pinecone DB
api_key = "828c0ba7-fbe7-4f81-bd61-5b9c8ae0912a"
# set Pinecone environment - find next to API key in console
env = "gcp-starter"
pinecone.init(
    api_key=api_key,
    environment=env
)
index_name='npc-rag'
index = pinecone.Index(index_name)


col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("""
        <img src="https://media4.giphy.com/media/13xxoHrXk4Rrdm/giphy.gif?cid=ecf05e47sr7dxi5kpveq2tcml2i43065zsgyyf9fkmual0l7&ep=v1_stickers_search&rid=giphy.gif&ct=s" width="150">
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div style='padding-top:30px'></div>", unsafe_allow_html=True)
    st.title("Team NPC")
    character = st.selectbox("Choose a character", 
                         character_names)
                         
                         
# Once the character is selected, get the bio
bio = list(character_level_dataset[character_level_dataset['name'] == character]['bio'])[0]

key = os.environ.get('key')
secret  = os.environ.get('secret')

session = boto3.Session(
    aws_access_key_id=key,
    aws_secret_access_key=secret,
    region_name = 'us-west-2'
)




def get_text(): 
    input_text = st.text_input("You: ","Hello", key="input")
    return input_text 

def query(payload):
    runtime = session.client('runtime.sagemaker')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload)
    result = json.loads(response['Body'].read().decode())
    return result

def generate_response(prompt):
    payload = {
        "inputs": prompt
    }

    response = query(json.dumps(payload))
    return response[0]["generated_text"]

def get_previous_responses(user_input):

    query_vec = embed_docs(user_input)
    
    res = index.query(query_vec, top_k=5, include_metadata=True)
    res = res['matches']
    res = [x['metadata']['text'] for x in res if x['score'] > 0.5]
    res = ' '.join(res)
    
    return res

# user_input = get_text()

# if user_input:
#     previous_res = get_previous_responses(user_input)
#     prompt = bio + " " + previous_res + " " + user_input
#     output = generate_response(prompt)
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)


# if st.session_state['generated']:
#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         message = st.session_state['generated'][i]  
#         st.write("Bot:", message)



# Input bar at the bottom
user_input = get_text()

# Extract character's name from the bio outside of the if block
character_name = list(character_level_dataset[character_level_dataset['name'] == character]['name'])[0]

if user_input:
    previous_res = get_previous_responses(user_input)
    prompt = bio + " " + previous_res + " " + user_input
    output = generate_response(prompt)
    
    # Display user input
    st.write(f"You: {user_input}")
    
    # Display bot response
    st.write(f"{character_name}:", output)

# Display chat history
for i in range(len(st.session_state['past']) - 2, -1, -2):
    msg_type, msg = st.session_state['past'][i], st.session_state['past'][i - 1]
    st.write(f"{msg_type}:", msg)



# if user_input:
#     previous_res = get_previous_responses(user_input)
#     prompt = bio + " " + previous_res + " " + user_input
#     output = generate_response(prompt)
#     st.session_state.past.append(("Bot:", output))
#     st.session_state.past.append(("You:", user_input))

# # Display generated response on top of the input bar
# if st.session_state['generated']:
#     for i in range(len(st.session_state['generated']) - 1, -1, -1):
#         message = st.session_state['generated'][i]
#         st.write("Bot:", message)

# # Display chat history
# # Display chat history
# for i in range(len(st.session_state['past']) - 1, -1, -2):
#     msg_type, msg = st.session_state['past'][i - 1], st.session_state['past'][i]
#     st.write(msg_type, msg)
# if 'input' not in st.session_state:
#     st.session_state['input'] = '' 

# prompt = st.text_input("You: ", st.session_state['input'], key='input')

# Retrieve the answer from pinecone db
# query_vec = embed_docs(prompt)

# get the answer using RAG
# res = index.query(query_vec, top_k=5, include_metadata=True)
# res = res['matches']
# res = [x['metadata']['text'] for x in res if x['score'] > 0.5]
# res = ' '.join(res)

# if prompt:
#     prompt = bio + ' ' + res + ' ' + prompt
#     output = generate_response(prompt)
#     st.session_state.past.append(st.session_state.input)
#     st.session_state.generated.append(output)

# if st.session_state['generated']:
#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         message = st.session_state['generated'][i] 
#         st.write("NPC:", message)

