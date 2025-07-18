import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Use Streamlit's caching to load the model only once
@st.cache_resource
def load_model():
    """
    Loads the fine-tuned model and tokenizer from Hugging Face Hub.
    """
    repo_name = "Mayank-22/Mayank-AI"
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    model = AutoModelForCausalLM.from_pretrained(repo_name)
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()

# Set up the Streamlit page
st.title("💊 Indian Medicines AI Assistant")
st.write("Enter the name of a medicine to get information about its uses.")

# Get user input
medicine_name = st.text_input("Enter medecine query:")

# Create a button to trigger the model
if st.button("Get Information"):
    if medicine_name:
        with st.spinner("Generating response..."):
            # Format the prompt
            prompt = f"Q: Give me precise response for the following query, retrive the medicine data from same row of the following query {medicine_name}?"

            # Tokenize and generate
# Inside your app.py file

# ... (rest of the code)

            # Tokenize and generate
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                  
                    max_new_tokens=100,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    # Add this line to fix the repetition
                    repetition_penalty=1.2 
                )
            
            # Decode and display the result
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # ... (rest of the code)
            
            st.subheader("Model Response:")
            st.write(result)
    else:
        st.warning("Please enter a medicine name.")
