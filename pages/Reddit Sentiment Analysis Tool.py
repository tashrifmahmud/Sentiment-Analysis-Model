import os
import praw
from dotenv import load_dotenv
import streamlit as st
from urllib.parse import urlparse, parse_qs
import time
import torch

# Tab name and icon
st.set_page_config(
    page_title="Reddit Sentiment Analysis", 
    page_icon="https://static-00.iconduck.com/assets.00/reddit-fill-logo-icon-2048x2048-lhgwkdq9.png",  
    layout="wide",  
    initial_sidebar_state="expanded"  
)

# Sidebar for links
with st.sidebar:
    st.header("About Reddit Tool:")
    st.markdown("\n- This tool analyzes Reddit comments using the Reddit API to fetch up to 100 comments from a post. It uses a custom pre-trained NLP model, fine-tuned on movie reviews, to classify comments as positive or negative. \n- The results include a sentiment breakdown and an overall summary, making it ideal for analyzing trends and opinions in Reddit discussions.")
    st.markdown("### :link: Links:\n- :cat: [GitHub](https://github.com/tashrifmahmud/Sentiment-Analysis-Model)\n- :hugging_face: [Hugging Face](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2)")
    st.markdown(":space_invader: Created by: [Tashrif Mahmud](https://www.linkedin.com/in/tashrifmahmud)")

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load environment variables from .env file
load_dotenv()

# Retrieve Reddit API credentials
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

# Initialize Reddit API connection
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# Streamlit interface
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
        <img src="https://www.iconpacks.net/icons/2/free-reddit-logo-icon-2436-thumb.png" alt="Reddit Logo" style="width: 100px; height: auto; margin-right: 20px;">
        <h1 style="margin: 0; font-size: 36px; color: #333;">Reddit Sentiment Analysis Tool</h1>
    </div>
    """,
    unsafe_allow_html=True
)


st.info("Analyze the sentiment of comments from a specific Reddit post!", icon="🎞")
# Load the sentiment analysis model
st.success(f"Using {'GPU' if device == 0 else 'CPU'} for inference.", icon="✈")
# Banner
st.image("https://i.imgur.com/g1JMYYr.png", use_container_width=True)

# Input field for Reddit post URL
post_url = st.text_input("Enter Reddit post URL:")
max_comments = st.number_input("Maximum number of comments to analyze (default: 50):", min_value=1, max_value=100, value=50, step=1)
run_button = st.button("Fetch and Analyze")

# Function to fetch comments from a Reddit post
def fetch_post_comments(url, max_comments):
    comments = []
    try:
        # Extract the submission ID from the URL
        path_parts = urlparse(url).path.split("/")
        if "comments" in path_parts:
            submission_id = path_parts[path_parts.index("comments") + 1]
        else:
            raise ValueError("Invalid Reddit post URL format.")
        
        submission = reddit.submission(id=submission_id)
        
        # Fetch all comments up to the maximum specified
        submission.comments.replace_more(limit=0)  # Flatten nested comments
        for comment in submission.comments.list():
            if len(comment.body.strip()) > 20:  # Filter out short comments
                comments.append(comment.body.strip())
                if len(comments) >= max_comments:
                    break
        return comments
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []

# Function to analyze sentiment using a sentiment analysis pipeline
def analyze_sentiment(comments):
    from transformers import pipeline

    # Load the sentiment analysis model
    sentiment_model = pipeline("text-classification", model="tashrifmahmud/sentiment_analysis_model_v2", device=device)

    results = []
    for comment in comments:
        result = sentiment_model(comment, truncation=True, batch_size=16)
        sentiment = result[0]['label']
        confidence = result[0]['score']
        results.append((comment, sentiment, confidence))
    return results

# Function to determine overall sentiment
def get_overall_sentiment(positive_count, negative_count):
    total = positive_count + negative_count
    if total == 0:
        return "No comments analyzed"

    positive_percentage = (positive_count / total) * 100
    negative_percentage = (negative_count / total) * 100

    if positive_percentage > 75:
        return "Overwhelmingly Positive"
    elif positive_percentage > 65:
        return "Very Positive"
    elif positive_percentage > 55:
        return "Moderately Positive"
    elif positive_percentage > 50:
        return "Slightly Positive"
    elif negative_percentage > 75:
        return "Overwhelmingly Negative"
    elif negative_percentage > 65:
        return "Very Negative"
    elif negative_percentage > 55:
        return "Moderately Negative"
    elif negative_percentage > 50:
        return "Slightly Negative"
    else:
        return "Mixed Emotions"

# Main logic
if run_button and post_url:
    st.write(f"Fetching comments from post: {post_url}")
    comments = fetch_post_comments(post_url, max_comments)

    if comments:
        st.info(f"🔍 Analyzing {len(comments)} comments...")
        results = analyze_sentiment(comments)

        positive_count = 0
        negative_count = 0

        # Count positive and negative comments
        for _, sentiment, _ in results:
            if sentiment == "POSITIVE":
                positive_count += 1
            else:
                negative_count += 1

        # Calculate sentiment summary
        overall_sentiment = get_overall_sentiment(positive_count, negative_count)
        total = positive_count + negative_count
        positive_percentage = (positive_count / total) * 100 if total > 0 else 0
        negative_percentage = (negative_count / total) * 100 if total > 0 else 0

        # Determine box color based on overall sentiment
        box_color = "#2ECC71" if positive_percentage > negative_percentage else "#E74C3C"

        # Display summary on top with dynamic color
        st.markdown(f"""
        <div style="
            text-align: center; 
            padding: 20px; 
            border: 2px solid #444; 
            border-radius: 10px; 
            background-color: {box_color}; 
            color: #FFFFFF;
        ">
            <h1 style="margin-bottom: 10px;">{overall_sentiment}</h1>
            <p><strong>Positive:</strong> {positive_count} ({positive_percentage:.2f}%)</p>
            <p><strong>Negative:</strong> {negative_count} ({negative_percentage:.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)

        # Display individual comment results in styled boxes
        for comment, sentiment, confidence in results:
            truncated_comment = comment[:60] + '...' if len(comment) > 60 else comment
            color = "#2ECC71" if sentiment == "POSITIVE" else "#E74C3C"
            st.markdown(f"""
            <div style="
                border: 1px solid {color}; 
                border-radius: 8px; 
                padding: 10px; 
                margin-bottom: 10px; 
                background-color: #f9f9f9; 
                display: flex; 
                justify-content: space-between;
                align-items: center;
            ">
                <div style="flex: 1; color: {color}; font-weight: bold; text-align: left;">{sentiment}</div>
                <div style="flex: 3; text-align: left; font-size: 14px; color: #333;">{truncated_comment}</div>
                <div style="flex: 1; text-align: right; font-size: 14px; color: #888;">Confidence: {confidence:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("No comments fetched. Please check the post URL or try again.")