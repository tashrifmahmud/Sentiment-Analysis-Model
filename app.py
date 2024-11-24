import streamlit as st
from transformers import pipeline

# Title
st.title(":bar_chart: Sentiment Analysis with Tashrif's LLM Model")
st.markdown(":cat: [GitHub Repository](https://github.com/tashrifmahmud/LLM-Project) | :hugging_face: [Hugging Face Model](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2)")

# Banner
st.image("https://exemplary.ai/img/blog/sentiment-analysis/sentiment-analysis.svg", use_column_width=True)

# Sidebar for links
with st.sidebar:
    st.header("More about this Project:")
    st.markdown("### :space_invader: Craeted by: Tashrif Mahmud\n- This model is a fine-tuned DistilBERT transformer for binary sentiment analysis. Initially trained on the IMDB dataset and later tuned with Rotten Tomatoes dataset, it distinguishes positive and negative text based movie reviews.")
    st.markdown("### :link: Links:\n- :cat: [GitHub](https://github.com/tashrifmahmud/LLM-Project)\n- :hugging_face: [Hugging Face](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2)")


# Load the sentiment analysis model
st.write(f"Using {'GPU' if device == 0 else 'CPU'} for inference.")
pipe = pipeline("text-classification", model="tashrifmahmud/sentiment_analysis_model_v2")

            
# Input text box
text_input = st.text_area("Enter text for sentiment analysis:")

# Buttons for user interaction
run_button = st.button("Run Prediction")
reset_button = st.button("Reset")

# State management: Run prediction or reset input
if run_button:
    if text_input.strip():
        result = pipe(text_input)
        st.write(f"Sentiment: {result[0]['label']}")
        st.write(f"Confidence: {result[0]['score']:.2f}")
        st.balloons()  # Fun animation
        st.success("Prediction complete!")
    else:
        st.warning("Please enter some text before running the prediction.")

if reset_button:
    st.experimental_rerun()  # Resets the app by reloading it
