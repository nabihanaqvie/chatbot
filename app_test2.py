import streamlit as st
import boto3
import requests
import json
import os


endpoint_name = "huggingface-pytorch-tgi-inference-2023-11-21-00-38-12-570" 


col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("![Alt Text](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExOWRuM3huajhrbHZvemVsazk0dDY1cWJtNWpkdXN6MXV3Mnd6ZGJuOSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jVflkGtOAL0qIrZ6nL/giphy.gif)")

with col2:
    st.title("Team NPC")
    character = st.selectbox("Choose a character", 
                         ["Elsa", "Indiana Jones", "Naruto Uzumaki", "Hermione Granger", "Sailor Moon", "Bugs Bunny"])

if 'generated' not in st.session_state: 
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


key = os.environ.get('key')
secret  = os.environ.get('secret')

session = boto3.Session(
    aws_access_key_id=key,
    aws_secret_access_key=secret,
    region_name = 'us-west-2'
)



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

if 'input' not in st.session_state:
    st.session_state['input'] = '' 

prompt = st.text_input("You: ", st.session_state['input'], key='input')

if prompt:
    output = generate_response(prompt)
    st.session_state.past.append(st.session_state.input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message = st.session_state['generated'][i] 
        st.write("NPC:", message)