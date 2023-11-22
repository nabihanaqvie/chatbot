import streamlit as st
import boto3
import requests
import json


endpoint_name = "huggingface-pytorch-tgi-inference-2023-11-21-00-38-12-570" 

st.title("Team NPC")

if 'generated' not in st.session_state: 
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []





def query(payload):
    region = 'us-west-2'
    runtime = boto3.client('runtime.sagemaker', region_name=region, use_ssl=False)
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
    return response

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