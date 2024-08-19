import streamlit as st
from transformers import pipeline
from PIL import Image

def generate_response(predictions):
    """
    Generate a response from the chatbot based on the classification results.

    Args:
        predictions (list): A list of dictionaries containing the classification results.

    Returns:
        str: A response string like "The plant is a [label] with [score]% confidence."
    """
    response = ""
    for p in predictions:
        response += f"The plant is a {p['label']} with {round(p['score'] * 100, 1)}% confidence.\n"
    return response

# Set page config
st.set_page_config(page_title="🤗💬 Garden Guru")

# Create a sidebar for accepting Hugging Face authentication credentials
# with st.sidebar:
#     st.title('🤗💬 HugChat')
#     if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
#         st.success('HuggingFace Login credentials already provided!', icon='✅')
#         hf_email = st.secrets['EMAIL']
#         hf_pass = st.secrets['PASS']
#     else:
#         hf_email = st.text_input('Enter E-mail:', type='password')
#         hf_pass = st.text_input('Enter password:', type='password')
#         if not (hf_email and hf_pass):
#             st.warning('Please enter your credentials!', icon='⚠️')
#         else:
#             st.success('Proceed to entering your prompt message!', icon='👉')
#     st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')

# Initialize chatbot session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Load Hugging Face model
pipeline = pipeline(task="image-classification", model="dima806/medicinal_plants_image_detection")

# Create a file uploader for the user to upload an image
file_name = st.file_uploader("Upload a plant image")

if file_name is not None:
    # Display the uploaded image
    image = Image.open(file_name)
    st.image(image, use_column_width=True)

    # Classify the image using the Hugging Face model
    predictions = pipeline(image)

    # Display the classification results
    st.header("Classification Results")
    for p in predictions:
        st.subheader(f"{p['label']}: {round(p['score'] * 100, 1)}%")

    # Generate a response from the chatbot
    response = generate_response(predictions)
    st.write(response)

    # Update the chatbot session state
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
