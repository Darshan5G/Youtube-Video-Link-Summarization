# import os
# from dotenv import load_dotenv
# import streamlit as st
# from langchain.document_loaders import YoutubeLoader
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import load_summarize_chain
# from langchain.schema import Document
# from gtts import gTTS
# import io
# from langchain_community.chat_models import ChatOllama

# # Load environment variables
# load_dotenv()
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")


# # Function to fetch available transcript languages
# def get_transcript_languages(youtube_video_url):
#     try:
#         video_id = youtube_video_url.split("v=")[-1].split("&")[0]
#         languages = YouTubeTranscriptApi.list_transcripts(video_id)
#         return [lang.language for lang in languages]
#     except TranscriptsDisabled:
#         return "Subtitles are disabled for this video."
#     except Exception as e:
#         return f"Error fetching transcript languages: {e}"


# # Function to generate a summary from the YouTube video
# def generate_summary(url):
#     select_lang = get_transcript_languages(url)[0][:2].lower()  
#     loader = YoutubeLoader.from_youtube_url(url, language=[select_lang], translation=select_lang)
#     docs = loader.load()

#     transcript = docs[0].page_content

#     # Split the transcript into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=len(transcript) // 5, chunk_overlap=50)
#     text_chunks = text_splitter.split_text(transcript)

#     # Convert text chunks into Document objects
#     final_documents = [Document(page_content=chunk) for chunk in text_chunks]

#     # Define the map prompt template
#     chunks_prompt = """
#     Please summarize the text based on the input text, language is not changed.
#     <text>
#     {text}
#     <text>
#     Summary:
#     """
#     map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

#     # Define the final prompt template
#     final_prompt = """
#     Make sure the summary is clear, easy to understand, and leaves out extra details.
#     Focus on the main ideas.
#     Final summary in 10 to 300 words based on video length and characters <length> and it is not written in output. Language is not changed
#     <text>
#     {text}
#     <text>
#     """
#     final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)

#     # Initialize the LLM
#     llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
    
#     # llm = ChatOllama(model='llama3.2')  

#     # Load the summarization chain
#     summary_chain = load_summarize_chain(
#         llm=llm,
#         chain_type="map_reduce",
#         map_prompt=map_prompt_template,
#         combine_prompt=final_prompt_template,
#         verbose=False
#     )

#     # Show the summarization chain on the final documents
#     summary = summary_chain.run(final_documents)
#     return summary, select_lang


# # Function to convert text summary into speech
# def speak_summary(summary, language):
#     myobj = gTTS(text=summary, lang=language, slow=False)
#     audio_fp = io.BytesIO()
#     myobj.write_to_fp(audio_fp)
#     audio_fp.seek(0)

#     return audio_fp


# # Streamlit UI setup
# st.set_page_config(page_title="YouTube Video Summarizer", page_icon="🎥", layout="wide")
# st.title("🎬 YouTube Video Summarizer & Speaker")
# st.markdown("Welcome to the **YouTube Video Summarizer & Speaker** tool! 😄")

# # Input the YouTube URL
# url = st.text_input("Enter YouTube Video URL:", key="video_url")
# # select_lang = get_transcript_languages(url)[0][:2].lower()

# # User interaction for generating summary
# if st.button("🔍 Generate Summary"):
#     if url:
#         with st.spinner(" Generating summary... 🔄"):
#             summary, language = generate_summary(url)
#             st.session_state.summary = summary  
#             st.session_state.language = language  
        
#         st.markdown("📜 **Video Summary**:")
#         st.write(summary)
#     else:
#         st.error("❗ Please enter a valid YouTube video URL.")

# # Button to speak the summary
# if st.button("🔊 Speak Summary"):
#     select_lang = get_transcript_languages(url)[0][:2].lower()
#     if 'summary' in st.session_state: 
#         audio_fp = speak_summary(st.session_state.summary, language)
#         st.audio(audio_fp, format="audio/mp3") 
#     else:
#         st.error("❗ Please generate a summary first.")






# import streamlit as st
# from gtts import gTTS
# import io

# # Function to convert text summary into speech
# def speak_summary(summary, language):
#     # Create a gTTS object
#     myobj = gTTS(text=summary, lang=language, slow=False)
    
#     # Save the audio in memory using BytesIO
#     audio_fp = io.BytesIO()
#     myobj.write_to_fp(audio_fp)  # Use write_to_fp instead of save
#     audio_fp.seek(0)  # Move the pointer to the start of the audio file
    
#     return audio_fp

# # Streamlit UI for "Speak Summary"
# st.set_page_config(page_title="Speak Summary", page_icon="🔊", layout="wide")
# st.title("🔊 Speak Summary Tool")

# # Input for the user to type or paste the summary
# summary = st.text_area("Enter Summary Text:", "", height=200)

# # Input for selecting the language
# language = st.selectbox("Select Language for Speech:", ["en", "es", "fr", "de", "it", "hi", "ja", "pt"])

# # Button to speak the summary
# if st.button("🔊 Speak Summary"):
#     if summary:
#         audio_fp = speak_summary(summary, language)
#         st.audio(audio_fp, format="audio/mp3")  # Play the audio
#     else:
#         st.error("❗ Please enter a summary to speak.")




import os
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
import re

# Load environment variables
# load_dotenv()
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")



# Function to validate the YouTube URL (supports both full and shortened URLs)
def is_valid_youtube_url(url):
    # Regex pattern to match both full and shortened YouTube URLs
    pattern = r"(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://(?:www\.)?youtu\.be/[\w-]+)"
    
    # Match the URL against the pattern
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

# Function to generate a summary from the YouTube video
def generate_summary(url):
    try:
        select_lang = get_transcript_languages(url)[0][:2].lower()  
        if isinstance(select_lang, str) and select_lang.startswith('Error'):
            return None, select_lang  # Error in fetching transcript

        loader = YoutubeLoader.from_youtube_url(url, language=[select_lang], translation=select_lang)
        docs = loader.load()

        if not docs:
            return None, "No transcript available for this video."

        transcript = docs[0].page_content

        # Split the transcript into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=len(transcript) // 5, chunk_overlap=50)
        text_chunks = text_splitter.split_text(transcript)

        # Convert text chunks into Document objects
        final_documents = [Document(page_content=chunk) for chunk in text_chunks]

        # Define the map prompt template
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

        # Define the final prompt template
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

        # Initialize the LLM
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])

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
