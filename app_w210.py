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

# Set environment variables

# Key and secret for sagemaker
key = os.environ.get('key')
secret  = os.environ.get('secret')

# Define the endpoints
dialog_studio_endpoint_name = "huggingface-pytorch-tgi-inference-2023-12-10-19-14-16-205"
llama_endpoint_name = "huggingface-pytorch-tgi-inference-2023-12-13-00-33-22-752"

# Connect to AWS Sagemaker session
session = boto3.Session(
    aws_access_key_id=key,
    aws_secret_access_key=secret,
    region_name = 'us-west-2'
)

# Connect to pinecone DB

# Set environment
api_key = "828c0ba7-fbe7-4f81-bd61-5b9c8ae0912a"
env = "gcp-starter"

# Initialize pinecone
pinecone.init(
    api_key=api_key,
    environment=env
)

# Connect to the index containing our custom RAG database for this project
index_name='npc-rag'
index = pinecone.Index(index_name)
                         

# Create the sidebar
st.sidebar.image("berkeley1.png", width=300)
st.sidebar.markdown("""<div style="text-align:center;">""", unsafe_allow_html=True) 
st.sidebar.title("W210 - Capstone")
st.sidebar.markdown("""[Sarah Hoover](https://www.linkedin.com/in/sarah-hoover-08816bba/), [Nabiha Naqvie](https://www.linkedin.com/in/nabiha-naqvie-22765612a/), [Bindu Thota](https://www.linkedin.com/in/bindu-thota/), and [Dave Zack](https://www.linkedin.com/in/dave-zack/)""")
st.sidebar.markdown("""</div>""", unsafe_allow_html=True)

st.sidebar.markdown("""---""")
st.sidebar.header("About")
st.sidebar.info("""
    In world of scripted NPCs(Non-Playable Character), Berkeley MIDS students joined forces to create LLM-Powered NPCs. NPCs are game story characters that primarily exist to provide important information regarding quests, health etc and develop the game world. Current NPCs embody one-directional, limited and non-replayable conversations that have gated the gaming experience. The motivation for this project is to enhance gaming experience through implementing LLM-powered NPCs that will allow for dynamic, multi-dimensional, and replayable NPC interactions. The MVP (Minimal Viable Product) provides 2 modes, interacting with an existing character or interacting with a character that you create yourself. It also demonstrates that game developers can achieve the creation of NPCs with less effort. In order to achieve LLM-Powered NPCs, the team contructed an ML pipeline that included both DialogStudio-T5 and LLaMa-2 implementation, with the enhancment of Retrieval Augmented Generation(RAG).

""")

st.sidebar.markdown("""---""")

st.sidebar.header("Source Code")
st.sidebar.markdown("""
    View the source code on [GitHub](https://github.com/nabihanaqvie/chatbot/blob/main/app_w210.py)  
""", unsafe_allow_html=True)
st.sidebar.markdown("""---""")


# Text box for chatting with the bot
def get_text():
    input_text = st.text_input("You: ","Hello", key="input")
    return input_text

# Radio button - make your own character or choose from existing characters
choice = st.radio(
    "What would you like to do?",
    ["Create your own character", "Choose an existing character"])

# Only need to read in the data if we're choosing from an existing character
if choice == 'Choose an existing character':

    # Import the dataset to get the character names/bios
    npc_train = load_dataset("amaydle/npc-dialogue", split="train")
    npc_test = load_dataset("amaydle/npc-dialogue", split="test")

    # Automatically splits it into train and test for you using all characters we've seen before, including those in test

    # First, transform them into pandas DFs
    train = pd.DataFrame(data = {'name': npc_train['Name'], 'bio':npc_train['Biography'], 'query':npc_train['Query'], 'response':npc_train['Response'], 'emotion':npc_train['Emotion']})
    test = pd.DataFrame(data = {'name': npc_test['Name'], 'bio':npc_test['Biography'], 'query':npc_test['Query'], 'response':npc_test['Response'], 'emotion':npc_test['Emotion']})

    # Now combine into a single df
    npc = pd.concat([train, test])

    # Create a character-level dataset, since the characters show up multiple times in the dataset
    character_level_dataset = npc[['name', 'bio']]
    character_level_dataset.drop_duplicates(inplace=True)

    # Get a list of just the names to populate the dropdown
    character_names = list(pd.unique(character_level_dataset['name']))
    
    # Create display
    col1, col2 = st.columns([1, 5])
    
    # Display character
    with col1:
        st.markdown("""
            <img src="https://media4.giphy.com/media/13xxoHrXk4Rrdm/giphy.gif?cid=ecf05e47sr7dxi5kpveq2tcml2i43065zsgyyf9fkmual0l7&ep=v1_stickers_search&rid=giphy.gif&ct=s" width="150">
        """, unsafe_allow_html=True)

    # Create character dropdown
    with col2:
        st.markdown("<div style='padding-top:30px'></div>", unsafe_allow_html=True)
        st.title("Team NPC")
        character = st.selectbox("Choose a character",
                             character_names)
                             
    # Once the character is selected, get the bio
    bio = list(character_level_dataset[character_level_dataset['name'] == character]['bio'])[0]

# If create your own character, different set up
else:

    # Input box for character name
    def create_character_name():
        input_text = st.text_input("Character name: ","Spongebob Squarepants", key="new_character")
        return input_text
    
    # Large input box for bio
    def create_bio():
        input_text = st.text_area("Bio: ", "A square yellow sponge named SpongeBob SquarePants lives in a pineapple with his pet snail, Gary, in the city of Bikini Bottom on the floor of the Pacific Ocean. He works as a fry cook at the Krusty Krab. During his time off, SpongeBob has a knack for attracting trouble with his starfish best friend, Patrick. Arrogant octopus Squidward Tentacles, SpongeBob’s neighbor, dislikes SpongeBob because of his childlike behavior.", key="new_bio")
        return input_text
    
    # Create display
    col1, col2 = st.columns([1, 5])
    
    # Show character
    with col1:
        st.markdown("""
            <img src="https://media4.giphy.com/media/13xxoHrXk4Rrdm/giphy.gif?cid=ecf05e47sr7dxi5kpveq2tcml2i43065zsgyyf9fkmual0l7&ep=v1_stickers_search&rid=giphy.gif&ct=s" width="150">
        """, unsafe_allow_html=True)

    # Inputs - character name and bio
    with col2:
        st.markdown("<div style='padding-top:30px'></div>", unsafe_allow_html=True)
        st.title("Team NPC")
        character = create_character_name()
        bio = create_bio()

# Now that we know the character, place input bar at the bottom to chat with that character
user_input = get_text()


# Set up the sentence-level encoder for RAG
rag_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to encode the input for retrieval
def embed_docs(docs: List[str]) -> List[List[float]]:
    # Use the sentence-level encoder to produce embeddings
    out = rag_encoder.encode(docs)
    return out.tolist()

# Save the current responses
if 'generated' not in st.session_state: 
    st.session_state['generated'] = []

# Create an opening to store the previous responses
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Clear memory after switching character
if character != st.session_state.get("current_character"):
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state["current_character"] = character

# Function to produce the output from Sagemaker
def query(payload, endpoint_name):
    runtime = session.client('runtime.sagemaker')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload)
    result = json.loads(response['Body'].read().decode())
    return result

# For dialogstudio, just need character name, bio, input from user
def generate_dialog_studio_response(prompt):
    # Set payload for sagemaker
    payload = {
        "inputs": prompt
    }
    # Get the generated dialog from dialogstudio
    response = query(json.dumps(payload), dialog_studio_endpoint_name)
    return response[0]["generated_text"]

# For llama-2, we need character name, bio, plus RAG results (if applicable), as well as input from user
def generate_llama2_response(prompt, character, bio):
    
    # Concatenate inputs to pass to sagemaker
    input = character + ' ' + bio + ' ' + prompt
    
    payload = {
        "inputs": input
    }
    
    print(input)
    
    # Get the response from sagemaker
    response = query(json.dumps(payload), llama_endpoint_name)
    full_response = response[0]["generated_text"]
    
    # Llama-2 will include the original query in the input- need to remove this from the output
    llama_response = full_response.split(prompt)[1]
    
    # Sometimes will continue having further conversations with itself - take just the first response
    llama_response = llama_response.split('.')[0]
    return llama_response

# Actually query the RAG db - using cosine similarity
def get_rag_responses(query_vec):
    res = index.query(query_vec, top_k=5, include_metadata=True)
    res = res['matches']
    res = [x['metadata']['text'] for x in res if x['score'] > 0.5]
    res = ' '.join(res)
    return res
    
# Query the bios as well - sentence by sentence- to see if there's any relevant info in there
def get_bio_responses(query_vec):
    # Code source: https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
    a = np.array(query_vec)
    # Separate the bio into sentences
    bios = bio.split('. ')
    # get the embeddings for each sentence separately
    bio_vecs = [embed_docs(x) for x in bios]
    # Get the cosine similarity of each sentence with the query
    cosines = [np.dot(a,b)/(norm(a)*norm(b)) for b in bio_vecs]
    # Return anything with a cosine similarity of at least 0.5
    cosines = [value for value in cosines if value > 0.5]
    return cosines

# If choosing an existing character, need to query then decide if using llama-2 or dialogstudio
if choice == 'Choose an existing character':
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
        
# If creating new character, always use llama-2
else:
    if user_input:
        query_vec = embed_docs(user_input)
        rag_results = get_rag_responses(query_vec)
        if len(rag_results) > 1:
            bio = bio + " " + rag_results
        output = generate_llama2_response(user_input, character, bio)
        st.session_state.past.append(("You", user_input))
        st.session_state.generated.append((character, output))

# Display chat history
for i in range(len(st.session_state['past']) - 1, -1, -1):
    msg_type, msg = st.session_state['past'][i]
    character_name, response = st.session_state['generated'][i]
    st.write(f"{msg_type}: {msg}")
    st.write(f"{character_name}: {response}")
