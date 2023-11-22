import streamlit as st
import boto3  
import json

endpoint_name = "your-endpoint"   

query_key = "query"   
next_key = "next"

def query(payload):
    runtime = boto3.client('runtime.sagemaker')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType='application/json',
        Body=payload)
    result = json.loads(response['Body'].read().decode())  
    return result

def generate_response(prompt):
    payload = {"inputs": prompt}
    response = query(json.dumps(payload))  
    return response 

def get_text():
    input_text = st.text_input("You: ", "", key=key) 
    return input_text

st.title("Chatbot")   

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = get_text(query_key)  

if input_text: 
    # Append history
    output = generate_response(user_input)
    st.session_state["history"].append({"message": user_input, "is_user": True})
    st.session_state["history"].append({"message": output, "is_user": False})

if st.session_state["history"]:
    # Display history 
    st.json(st.session_state["history"])
    
get_text(next_key) # Next input box