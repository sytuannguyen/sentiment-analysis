import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# List of pre-trained models for users to select
pretrained_models = {
    "Sentiment Analysis": "text-sentiment-analysis",
    "Text Classification (roberta-base)": "text-classification/roberta-base",
    "Text Classification (distilbert-base)": "text-classification/distilbert-base-cased",
    "Custom Model": "custom"
}

# Streamlit UI components
st.title("Hugging Face Transformers App")

# User input for choosing a pre-trained model or custom model
selected_model = st.selectbox("Select a pre-trained model:", list(pretrained_models.keys()))
user_input = st.text_area("Enter a sentence:", "")

if selected_model == "Sentiment Analysis":
    # Load the sentiment analysis model
    analyzer = pipeline('sentiment-analysis')

    if user_input:
        with st.spinner("Analyzing sentiment..."):
            sentiment = analyzer(user_input)
            predicted_sentiment = sentiment[0]['label']
            confidence = sentiment[0]['score']

            st.subheader("Predicted Sentiment:")
            st.write(predicted_sentiment)
            st.subheader("Confidence:")
            st.write(confidence)

elif selected_model.startswith("Text Classification"):
    # Load the selected text classification model
    model_name = pretrained_models[selected_model]
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if user_input:
        with st.spinner("Classifying text..."):
            inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax().item()

            st.subheader("Predicted Class (Label ID):")
            st.write(predicted_class)

else:  # Custom Model
    # User input for specifying a custom model and tokenizer
    custom_model_name = st.text_input("Enter a custom model name:", "")
    custom_tokenizer_name = st.text_input("Enter a custom tokenizer name:", "")

    if custom_model_name and custom_tokenizer_name:
        # Load the custom model and tokenizer
        custom_model = AutoModelForSequenceClassification.from_pretrained(custom_model_name)
        custom_tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer_name)

        if user_input:
            with st.spinner("Running custom model..."):
                inputs = custom_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
                outputs = custom_model(**inputs)
                logits = outputs.logits
                predicted_class = logits.argmax().item()

                st.subheader("Predicted Class (Label ID):")
                st.write(predicted_class)

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
