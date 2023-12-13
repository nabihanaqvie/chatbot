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
from numpy.linalg import norm

dialog_studio_endpoint_name = "huggingface-pytorch-tgi-inference-2023-12-10-19-14-16-205"
llama_endpoint_name = "huggingface-pytorch-tgi-inference-2023-12-13-00-33-22-752"

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


def get_text():
    input_text = st.text_input("You: ","Hello", key="input")
    return input_text

# Radio button - make your own character or choose from existing characters
choice = st.radio(
    "What would you like to do?",
    ["Create your own character", "Choose an existing character"])


if choice == 'Choose an existing character':

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

else:
    def create_character_name():
        input_text = st.text_input("Character name: ","Spongebob Squarepants", key="new_character")
        return input_text
        
    def create_bio():
        input_text = st.text_area("Bio: ", "A square yellow sponge named SpongeBob SquarePants lives in a pineapple with his pet snail, Gary, in the city of Bikini Bottom on the floor of the Pacific Ocean. He works as a fry cook at the Krusty Krab. During his time off, SpongeBob has a knack for attracting trouble with his starfish best friend, Patrick. Arrogant octopus Squidward Tentacles, SpongeBob’s neighbor, dislikes SpongeBob because of his childlike behavior.", key="new_bio")
        return input_text
    
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("""
            <img src="https://media4.giphy.com/media/13xxoHrXk4Rrdm/giphy.gif?cid=ecf05e47sr7dxi5kpveq2tcml2i43065zsgyyf9fkmual0l7&ep=v1_stickers_search&rid=giphy.gif&ct=s" width="150">
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='padding-top:30px'></div>", unsafe_allow_html=True)
        st.title("Team NPC")
        character = create_character_name()
        bio = create_bio()

# Input bar at the bottom
user_input = get_text()


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
                         
                        
key = os.environ.get('key')
secret  = os.environ.get('secret')

session = boto3.Session(
    aws_access_key_id=key,
    aws_secret_access_key=secret,
    region_name = 'us-west-2'
)

if 'generated' not in st.session_state: 
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if character != st.session_state.get("current_character"):
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state["current_character"] = character

def query(payload, endpoint_name):
    runtime = session.client('runtime.sagemaker')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload)
    result = json.loads(response['Body'].read().decode())
    return result

def generate_dialog_studio_response(prompt):
    payload = {
        "inputs": prompt
    }

    response = query(json.dumps(payload), dialog_studio_endpoint_name)
    return response[0]["generated_text"]

def generate_llama2_response(prompt, character, bio):
    
    input = character + ' ' + bio + ' ' + prompt
    
    payload = {
        "inputs": input
    }
    
    print(input)
    
    response = query(json.dumps(payload), llama_endpoint_name)
    
    full_response = response[0]["generated_text"]
    llama_response = full_response.split(prompt)[1]
    # take only first sentence
    llama_response = llama_response.split('.')[0]
    return llama_response

def get_rag_responses(query_vec):
    
    res = index.query(query_vec, top_k=5, include_metadata=True)
    res = res['matches']
    res = [x['metadata']['text'] for x in res if x['score'] > 0.5]
    res = ' '.join(res)
    
    return res
    
def get_bio_responses(query_vec):
    bio_vec = embed_docs(bio)
    # Code source: https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
    a = np.array(query_vec)
    bios = bio.split('. ')
    bio_vecs = [embed_docs(x) for x in bios]
    cosines = [np.dot(a,b)/(norm(a)*norm(b)) for b in bio_vecs]
    print(cosines)
    cosines = [value for value in cosines if value > 0.5]
    return cosines


if user_input:
    query_vec = embed_docs(user_input)
    rag_results = get_rag_responses(query_vec)
    print(get_bio_responses(query_vec))
    bio_responses = get_bio_responses(query_vec)
    if len(rag_results) > 1 or len(bio_responses) > 0:
        bio = bio + " " + rag_results
        output = generate_llama2_response(user_input, character, bio)
    else:
        output = generate_dialog_studio_response(character + " " + bio + " " + user_input)
    st.session_state.past.append(("You", user_input))
    st.session_state.generated.append((character, output))

# Display chat history
for i in range(len(st.session_state['past']) - 1, -1, -1):
    msg_type, msg = st.session_state['past'][i]
    character_name, response = st.session_state['generated'][i]
    st.write(f"{msg_type}: {msg}")
    st.write(f"{character_name}: {response}")
