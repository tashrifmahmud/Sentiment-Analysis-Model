import streamlit as st
import torch
from transformers import pipeline

# Title
st.title(":bar_chart: Sentiment Analysis with LLM")
st.markdown("[GitHub Repository](https://github.com/tashrifmahmud/Sentiment-Analysis-Model) | [Hugging Face Model](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2)")

# Banner
st.image("https://www.qdegrees.com/uploads/blogs-img/sentiment-analysis-and-overview.jpg", use_container_width=True)

# Sidebar for links
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; padding: 10px; background-color: #eef2f5; border-radius: 8px; border-left: 5px solid #007bff;">
        <img src="https://www.iconpacks.net/icons/2/free-reddit-logo-icon-2436-thumb.png" alt="Reddit Logo" style="width: 40px; height: 40px; margin-right: 10px;">
        <div style="font-size: 16px; color: #333;">Try out the new <strong>Reddit Tool</strong>!</div>
    </div>
    """, unsafe_allow_html=True)

    st.header("More about this Project:")
    st.markdown("This model is a finetuned DistilBERT transformer for binary sentiment analysis. Initially trained on the IMDB dataset and later tuned with Rotten Tomatoes dataset, it distinguishes positive and negative text-based movie reviews.")
    st.markdown("### :link: Links:\n- :cat: [GitHub](https://github.com/tashrifmahmud/Sentiment-Analysis-Model)\n- :hugging_face: [Hugging Face](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2)")
    st.markdown(":space_invader: Created by: [Tashrif Mahmud](https://www.linkedin.com/in/tashrifmahmud)")

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load the sentiment analysis model
st.success(f"Using {'GPU' if device == 0 else 'CPU'} for inference.", icon="✈")
pipe = pipeline("text-classification", model="tashrifmahmud/sentiment_analysis_model_v2", device=device)

# Initialize session state for text input
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# Function to handle reset button
def reset_input():
    st.session_state.text_input = ""  # Clear the input via callback

st.info("Enter text for sentiment analysis:", icon="⬇")

# Input text box
text_input = st.text_area("Text Box:", value=st.session_state.text_input, key="text_input")

# Buttons for user interaction
run_button = st.button("Run Prediction")
reset_button = st.button("Reset", on_click=reset_input)  # Attach callback to reset

# State management: Run prediction
if run_button:
    if text_input.strip():
        result = pipe(text_input)
        
        # Highlight the result with custom styling
        sentiment = result[0]['label']
        confidence = result[0]['score']

        # Display Sentiment and Confidence with background color for better visibility
        if sentiment == 'POSITIVE':  # Positive sentiment
            sentiment_color = "#2dbe3e"  # Green for positive
            sentiment_text = "😊 Positive"
        elif sentiment == 'NEGATIVE':  # Negative sentiment
            sentiment_color = "#ff6f61"  # Red for negative
            sentiment_text = "😡 Negative"
        else:
            sentiment_color = "#d3d3d3"  # Grey for unknown (fallback)
            sentiment_text = "Unknown"

        st.markdown(f"""
        <div style="background-color:{sentiment_color}; padding: 10px; border-radius: 5px; text-align: center;">
            <h2 style="color: white;">Sentiment: {sentiment_text}</h2>
            <h3 style="color: white;">Confidence: {confidence:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()  # Fun animation
        st.success("Prediction complete!")
    else:
        st.warning("Please enter some text before running the prediction.", icon="⚠️")
