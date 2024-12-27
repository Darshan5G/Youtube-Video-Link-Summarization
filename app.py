import os
import re
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import load_summarize_chain
from langchain.schema import Document
from gtts import gTTS
import io
from langchain_community.chat_models import ChatOllama

# Load environment variables (not needed on Streamlit Cloud if using secrets)
# load_dotenv()
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")

groq_api_key = st.secrets["GROQ_API_KEY"]["value"]  
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]["value"]

# Function to Validate the YouTube URL.
def is_valid_youtube_url(url):
    pattern = r"(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://(?:www\.)?youtu\.be/[\w-]+)"
    return bool(re.match(pattern, url))

# Function to get transcript languages
def get_transcript_languages(youtube_video_url):
    try:
        video_id_match = re.search(r"(?:v=|/)([a-zA-Z0-9_-]{11})(?=\b|/|$)", youtube_video_url)
        if not video_id_match:
            return "Invalid YouTube URL format."
        video_id = video_id_match.group(1)
        languages = YouTubeTranscriptApi.list_transcripts(video_id)
        return [lang.language for lang in languages]
    except TranscriptsDisabled:
        return "Subtitles are disabled for this video."
    except Exception as e:
        return f"Error fetching transcript languages: {e}"

# Function to select the auto-generated language
def select_auto_generated_language(url):
    try:
        languages = get_transcript_languages(url)
        if isinstance(languages, str):
            return languages  
        auto_generated_languages = [lang for lang in languages if "auto-generated" in lang.lower()]
        if not auto_generated_languages:
            return "No auto-generated transcript found."
        selected_lang = auto_generated_languages[0].split(" ")[0].lower()
        return selected_lang
    except Exception as e:
        return f"Error: {e}"

# Function to generate a summary from the YouTube video
def generate_summary(url):
    select_lang = select_auto_generated_language(url)[:2]

    # Load YouTube video using language selection
    loader = YoutubeLoader.from_youtube_url(url, language=[select_lang], translation=select_lang)
    docs = loader.load()

    # Extract transcript and metadata
    transcript = docs[0].page_content

    # Split the transcript into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=len(transcript) // 5, chunk_overlap=50)
    text_chunks = text_splitter.split_text(transcript)

    # Convert text chunks into Document objects
    final_documents = [Document(page_content=chunk) for chunk in text_chunks]

    # Define the map prompt template
    chunks_prompt = """
    Please summarize the text based on the input text, language is not changed.
    <text>
    {text}
    <text>
    Summary:
    """
    map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

    # Define the final prompt template
    final_prompt = """
    Make sure the summary is clear, easy to understand, and leaves out extra details.
    Focus on the main ideas.
    Final summary in 10 to 300 words based on video length and characters <length> and it is not written in output. Language is not changed
    <text>
    {text}
    <text>
    """
    final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)

    # Initialize the LLM
    llm = ChatGroq(model="distil-whisper-large-v3-en", groq_api_key=groq_api_key)

    # Load the summarization chain
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=final_prompt_template,
        verbose=False
    )

    # Show the summarization chain on the final documents
    summary = summary_chain.run(final_documents)
    return summary, select_lang

# Function to convert text summary into speech
def speak_summary(summary, language):
    # Check if the language is supported by gTTS
    supported_langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh']
    if language not in supported_langs:
        language = 'en'  # Default to English if not supported
    
    myobj = gTTS(text=summary, lang=language, slow=False)
    audio_fp = io.BytesIO()
    myobj.write_to_fp(audio_fp)
    audio_fp.seek(0)

    return audio_fp

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
