import streamlit as st
import subprocess
import json

def run_curl_command(input_text):
    # Construct the curl command
    prompt_text = f"summarize this: {input_text}"  # Combine the text into one prompt
    curl_command = [
        'curl', 'http://localhost:11434/api/generate',
        '-d', json.dumps({
            "model": "mistral",
            "prompt": prompt_text,
            "stream": True
        }),
        '-H', 'Content-Type: application/json'
    ]
    
    # Run the curl command and capture the output
    result = subprocess.run(curl_command, capture_output=True, text=True)
    
    # Process the result and extract only "response" fields
    responses = []
    for line in result.stdout.splitlines():
        try:
            json_data = json.loads(line)  # Parse each line as JSON
            if "response" in json_data:
                responses.append(json_data["response"])
        except json.JSONDecodeError:
            pass  # Ignore lines that can't be parsed as JSON

    # Join the responses into a single string
    return " ".join(responses)

# Streamlit interface
st.title("...")

# Input field for user
input_text = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if input_text:
        st.write("Generating response...")
        output = run_curl_command(input_text)
        st.text_area("Response", value=output, height=300)
    else:
        st.error("Please enter a prompt!")
