import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# List of pre-trained models for users to select
pretrained_models = {
    "Sentiment Analysis": "text-sentiment-analysis",
    "Text Classification": "Text Classification"
}

# Streamlit UI components
st.title("Social Sentiment Analysis")
# Subtitle showing author name and institution
st.markdown("**Dr. Tuan Nguyen-Sy** ")
st.markdown("<font color='gray'>Institute for Computational Science and Artificial Intelligence, Van Lang University</font>", unsafe_allow_html=True)

# User input for choosing a pre-trained model or custom model
selected_model = st.selectbox("Select a pre-trained model:", list(pretrained_models.keys()))
user_input = st.text_area("Enter a sentence:", "")

if selected_model == "Sentiment Analysis":
    # Load the sentiment analysis model
    analyzer = pipeline('sentiment-analysis')

elif selected_model == "Text Classification":
    # Load the sentiment analysis model
    analyzer = pipeline('text-classification')

if user_input:
    with st.spinner("Analyzing sentiment..."):
        sentiment = analyzer(user_input)
        predicted_sentiment = sentiment[0]['label']
        confidence = sentiment[0]['score']

        st.subheader("Predicted Sentiment:")
        st.write(predicted_sentiment)
        st.subheader("Confidence:")
        st.write(confidence)
        
# Instructions for the user
st.markdown(
    """
    **Instructions:**
    - Select a pre-trained model from the dropdown list.
    - Enter a sentence in the text box.
    - The app will perform the selected task based on the chosen model.
    - For the Custom Model option, you can specify your custom model and tokenizer names.
    """
)
