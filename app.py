import streamlit as st
from transformers import pipeline

# Load the sentiment analysis model
analyzer = pipeline('sentiment-analysis')

# Main function
def main():
  # Streamlit UI components
  st.title("Sentiment Analysis App")
  # Subtitle showing author name and institution
  st.markdown("**Dr. Tuan Nguyen-Sy** ")
  st.markdown("<font color='gray'>Institute for Computational Science and Artificial Intelligence, Van Lang University</font>", unsafe_allow_html=True)
  
  user_input = st.text_area("Enter a sentence:", "")
  
  # Perform sentiment analysis when the user submits the input
  if user_input:
      with st.spinner("Analyzing sentiment..."):
          # Analyze sentiment for the user input
          sentiment = analyzer(user_input)
          predicted_sentiment = sentiment[0]['label']
          confidence = sentiment[0]['score']
  
          # Display the predicted sentiment and confidence
          st.subheader("Predicted Sentiment:")
          st.write(predicted_sentiment)
          st.subheader("Confidence:")
          st.write(confidence)
  
  # Instructions for the user
  st.markdown(
      """
      **Instructions:**
      - Enter a sentence in the text box.
      - The app will predict the sentiment of the input sentence.
      """
  )

# Run the app
if __name__ == "__main__":
  main()
