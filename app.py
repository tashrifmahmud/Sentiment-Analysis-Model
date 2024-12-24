import streamlit as st
import torch
from transformers import pipeline

# Tab 
st.set_page_config(
    page_title="Sentiment Analysis Model", 
    page_icon="https://i.imgur.com/vScON4I.png",  
    layout="wide",  
    initial_sidebar_state="expanded"  
)

# Header
st.markdown(
    """
    <div style="
        display: flex; 
        align-items: center; 
        justify-content: center; 
        background-color: #f0f0f0; 
        padding: 20px; 
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    ">
        <img src="https://i.imgur.com/vScON4I.png" alt="Logo" style="width: 100px; height: auto; margin-right: 20px;">
        <h1 style="margin: 0; font-size: 36px; color: #333;">Sentiment Analysis with LLM</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# Links
st.markdown(
    """
    <div style="text-align: center;">
        <a href="https://github.com/tashrifmahmud/Sentiment-Analysis-Model" target="_blank" style="margin-right: 20px;">GitHub Repository</a>
        <a href="https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2" target="_blank">Hugging Face Model</a>
    </div>
    """,
    unsafe_allow_html=True
)

# Banner
st.image("https://i.imgur.com/1h8Pxfz.png", use_container_width=True)

# Sidebar
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

# Reset button
def reset_input():
    st.session_state.text_input = "" 

st.info("Enter text for sentiment analysis:", icon="⬇")

# Input text box
text_input = st.text_area("Text Box:", value=st.session_state.text_input, key="text_input")

# Buttons
run_button = st.button("Run Prediction")
reset_button = st.button("Reset", on_click=reset_input) 

# Run prediction
if run_button:
    if text_input.strip():
        result = pipe(text_input)
        
        sentiment = result[0]['label']
        confidence = result[0]['score']

        if sentiment == 'POSITIVE': 
            sentiment_color = "#2dbe3e"  
            sentiment_text = "😊 Positive"
        elif sentiment == 'NEGATIVE':  
            sentiment_color = "#ff6f61" 
            sentiment_text = "😡 Negative"
        else:
            sentiment_color = "#d3d3d3" 
            sentiment_text = "Unknown"

        st.markdown(f"""
        <div style="background-color:{sentiment_color}; padding: 10px; border-radius: 5px; text-align: center;">
            <h2 style="color: white;">Sentiment: {sentiment_text}</h2>
            <h3 style="color: white;">Confidence: {confidence:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.toast("Text analysis is done, scroll for results!", icon="✅") 
        st.success("Prediction complete!")
    else:
        st.warning("Please enter some text before running the prediction.", icon="⚠️")
