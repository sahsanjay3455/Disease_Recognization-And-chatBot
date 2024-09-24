import streamlit as st
from PIL import Image
import os

from streamlit_option_menu import option_menu
from gemini_utility import (load_gemini_pro_model,
                            gemini_pro_response,
                            gemini_pro_vision_response,
                            embeddings_model_response)

working_dir = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(
    page_title="Gemini AI",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    selected = option_menu('Gemini AI',
                           ['ChatBot',
                            'Disease Recognition',
                            'Ask me anything'],
                           menu_icon='robot', icons=['chat-dots-fill', 'prescription2', 'textarea-t', 'patch-question-fill'],
                           default_index=0
                           )
# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role
# chatbot page
if selected == 'ChatBot':
    model = load_gemini_pro_model()

   # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:  # Renamed for clarity
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chatbot's title on the page
    st.title("🤖 ChatBot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Gemini-Pro...")  # Renamed for clarity
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)  # Renamed for clarity

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)


# Image captioning page
if selected == "Disease Recognition":

    st.title("📷 An application that can help users to identify medical images")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption"):
        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((1000, 500))
            st.image(resized_img)

        default_prompt = """
        
        As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining medical images for a renowned hospital. Your expertise is crucial in identifying any anomalies, diseases, or health issues that may be present in the images.

Your Responsibilities include:

1. Detailed Analysis:Thoroughly analyze each image, focusing on identifying any abnormal findings.

2. Findings Report:Document all observed anomalies or signs of disease. Clearly articulate these findings in a structured format.

3. Recommendations and Next Steps: Based on your analysis, suggest potential next steps, including further tests or treatments as applicable.

4. Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.

Important Notes:

1. Scope of Response:Only respond if the image pertains to human health issues.

2. Clarity of Image:In cases where the image quality impedes clear analysis, note that certain aspects are 'Unable to be determined based on the provided image.'

 3. Disclaimer:Accompany your analysis with the disclaimer: "Consult with a Doctor before making any decisions."

4. Your insights are invaluable in guiding clinical decisions. Please proceed with the analysis, adhering to the structured approach outlined above

5.provide me an output response with these 4 heading  Detailed Analysis,Findings Report,Recommendations and Next Steps,Treatment Suggestions

6.Suggest some medicine name for temporary time  according to problem .
7.Conclusion:
Conclusion should me provided at end of the file.
        
        """
                      # change this prompt as per your requirement

        # get the caption of the image from the gemini-pro-vision LLM
        caption = gemini_pro_vision_response(default_prompt, image)

        with col2:
            st.info(caption)


# text embedding model
# if selected == "Embeded text":

#     st.title("🔡 Embeded Text")

#     # text box to enter prompt
#     user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

#     if st.button("Get Response"):
#         response = embeddings_model_response(user_prompt)
#         st.markdown(response)


# text embedding model

if selected == "Ask me anything":

    st.title("❓ Ask me a question")

    # text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response"):
        response = gemini_pro_response(user_prompt)
        st.markdown(response)
