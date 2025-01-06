import os
import re
import streamlit as st
import requests
from dotenv import load_dotenv


# Set environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# API URLs
API_URL = "http://127.0.0.1:8000"

# Function to validate the YouTube URL
def is_valid_youtube_url(url):
    pattern = r"(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://(?:www\.)?youtu\.be/[\w-]+)"
    return bool(re.match(pattern, url))

# Streamlit UI setup
st.set_page_config(page_title="YouTube Video Link Summarizer", page_icon="🎥", layout="wide")
st.title("🎬 YouTube Video Link Summarizer")

# Input the YouTube URL
url = st.text_input("Enter YouTube Video URL:", key="video_url")

# User interaction for generating summary
if st.button("🔍 Generate Summary"):
    if url:
        if is_valid_youtube_url(url):
            with st.spinner("Generating summary... 🔄"):
                response = requests.post(f"{API_URL}/generate_summary/", json={"url": url})
                if response.status_code == 200:
                    summary = response.json()['summary']
                    language = response.json()['language']
                    st.session_state.summary = summary
                    st.session_state.language = language
                    st.markdown("📜 **Video Summary**:")
                    st.write(summary)
                else:
                    st.error(f"❗ Error: {response.json().get('detail')}")
        else:
            st.warning("❗ Please provide a valid YouTube video URL.")
    else:
        st.warning("❗ Please provide a valid YouTube video URL.")

# Button to speak the summary
if st.button("🔊 Speak Summary"):
    if 'summary' in st.session_state:
        response = requests.get(f"{API_URL}/speak")
        if response.status_code == 200:
            audio_url = response.content
            st.audio(audio_url, format="audio/mp3")
        else:
            st.error("❗ Error generating audio.")
    else:
        st.warning("❗ Please generate a summary first.")
