import os
import time
import re
import io
from dotenv import load_dotenv
from gtts import gTTS
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import load_summarize_chain
from langchain.schema import Document
from langchain_groq import ChatGroq

# Load environment variables (ensure that .env file contains valid keys for local testing)
# if os.path.exists('.env'):
#     load_dotenv()

# Streamlit secrets for Streamlit Cloud
groq_api_key = st.secrets["GROQ_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]

# Function to validate the YouTube URL (supports both full and shortened URLs)
def is_valid_youtube_url(url):
    pattern = r"(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://(?:www\.)?youtu\.be/[\w-]+)"
    return bool(re.match(pattern, url))

# Function to fetch available transcript languages
def get_transcript_languages(youtube_video_url):
    try:
        video_id = youtube_video_url.split("v=")[-1].split("&")[0] if "youtube.com" in youtube_video_url else youtube_video_url.split("/")[-1]
        languages = YouTubeTranscriptApi.list_transcripts(video_id)
        return [lang.language for lang in languages]
    except TranscriptsDisabled:
        return "Subtitles are disabled for this video."
    except Exception as e:
        return f"Error fetching transcript languages: {e}"

# Retry function to handle transient errors (503 errors)
def retry_request(func, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt+1} failed, retrying in {delay} seconds... Error: {e}")
                time.sleep(delay)
            else:
                print(f"All retry attempts failed. Error: {e}")
                raise e

# Function to generate a summary from the YouTube video
def generate_summary(url):
    try:
        # Get the language for the transcript (assuming it's available)
        select_lang = get_transcript_languages(url)[0][:2].lower()
        if isinstance(select_lang, str) and select_lang.startswith('Error'):
            return None, select_lang  # Error in fetching transcript
        
        # Load the YouTube transcript using the selected language
        loader = YoutubeLoader.from_youtube_url(url, language=[select_lang], translation=select_lang)
        docs = loader.load()

        if not docs:
            return None, "No transcript available for this video."

        # Split the transcript into chunks
        transcript = docs[0].page_content
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=len(transcript) // 5, chunk_overlap=50)
        text_chunks = text_splitter.split_text(transcript)
        final_documents = [Document(page_content=chunk) for chunk in text_chunks]

        # Set up the prompt template for map-reduce summarization
        chunks_prompt = """
        You are a summarization model using a map-reduce approach. 
        Task is to summarize the text provided below. 
        You do not change the language of the text. 
        Focus only on creating a clear and concise summary while maintaining the original meaning.
        <text>
        {text}
        <text>
        Summary:
        """
        map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

        final_prompt = """
        You are a summarization model using the map-reduce approach. 
        Your task is to create a final summary from the text provided below. 

        Make sure the summary:
        - Is clear and easy to understand.
        - Focuses on the main ideas and leaves out unnecessary details.
        - Is between 10 to 300 words, depending on the length of the original content.
        - Does not change the language of the text.

        <text>
        {text}
        <text>
        """
        final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)

        # Initialize the LLM (Groq)
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

        # Retry the summarization chain in case of a service disruption
        summary_chain = retry_request(lambda: load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=final_prompt_template,
            verbose=False
        ))

        # Run the summarization chain on the final documents
        summary = summary_chain.run(final_documents)
        return summary, select_lang
    except Exception as e:
        return None, f"Error generating summary: {e}"

# Function to convert text summary into speech
def speak_summary(summary, language):
    myobj = gTTS(text=summary, lang=language, slow=False)
    audio_fp = io.BytesIO()
    myobj.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# Streamlit UI setup
st.set_page_config(page_title="YouTube Video Summarizer", page_icon="🎥", layout="wide")
st.title("🎬 YouTube Video Summarizer & Speaker")
st.markdown("Welcome to the **YouTube Video Summarizer & Speaker** tool! 😄")

# Input the YouTube URL
url = st.text_input("Enter YouTube Video URL:", key="video_url")

# User interaction for generating summary
if st.button("🔍 Generate Summary"):
    if url:
        if is_valid_youtube_url(url):
            with st.spinner("Generating summary... 🔄"):
                summary, language = generate_summary(url)
                if summary:
                    st.session_state.summary = summary
                    st.session_state.language = language
                    st.markdown("📜 **Video Summary**:")
                    st.write(summary)
                else:
                    st.error(f"❗ Error: {language}")
        else:
            st.warning("❗ Please provide a valid YouTube video URL.")
    else:
        st.warning("❗ Please provide a valid YouTube video URL.")

# Button to speak the summary
if st.button("🔊 Speak Summary"):
    if url:
        if 'summary' in st.session_state:
            audio_fp = speak_summary(st.session_state.summary, st.session_state.language)
            if audio_fp:
                st.audio(audio_fp, format="audio/mp3")
            else:
                st.error("❗ Error generating audio.")
        else:
            st.warning("❗ Please generate a summary first.")
    else:
        st.warning("❗ Please provide a valid YouTube video URL.")
