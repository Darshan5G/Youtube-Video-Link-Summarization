#Import All Required Libraries
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
import re

#Create the class for Youtube video link summary generator
class YouTubeSummaryGenerator:
    """
    A class to generate summary for YouTube videos and convert text summaries into speech.
    """
    
    def __init__(self, groq_api_key):
        """
        Initialize class with the Groq API key for access LLM models.
        """
        self.groq_api_key = groq_api_key


    def is_valid_youtube_url(self, url):
        """
        Validate the YouTube URL.
        """
        pattern = r"(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://(?:www\.)?youtu\.be/[\w-]+)"
        return bool(re.match(pattern, url))


    def get_transcript_languages(self, youtube_video_url):
        """
        Fetch available transcript languages for a given YouTube video URL.
        """
        try:
            video_id = youtube_video_url.split("v=")[-1].split("&")[0] if "youtube.com" in youtube_video_url else youtube_video_url.split("/")[-1]
            languages = YouTubeTranscriptApi.list_transcripts(video_id)
            return [lang.language for lang in languages]
        except TranscriptsDisabled:
            return "Subtitles are disabled for this video."
        except Exception as e:
            return f"Error fetching transcript languages: {e}"


    def generate_summary(self, url):
        """
        Generate a summary from the YouTube video transcript in same language.
        """
        try:
            select_lang = self.get_transcript_languages(url)[0][:2].lower()
            if isinstance(select_lang, str) and select_lang.startswith('Error'):
                return None, select_lang  

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
            Do not change the language of the text. 
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
            Task is to create a final summary from the text provided below. 

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
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=self.groq_api_key)

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


    def speak_summary(self, summary, language):
        """
        Convert the summary text to speech using gTTS (Google Text-to-Speech).
        """
        myobj = gTTS(text=summary, lang=language, slow=False)
        audio_fp = io.BytesIO()
        myobj.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp


#Create the class for user interface using streamlit
class YouTubeApp:
    """
    A class to handle the Streamlit interface for YouTube video summarization and speech generation.
    """
    
    def __init__(self, youtube_summary_generator):
        """
        Initialize the class with the YouTubeSummaryGenerator instance.
        """
        self.youtube_summary_generator = youtube_summary_generator


    def run(self):
        """
        Run the Streamlit app for YouTube video summarization and speech generation.
        """

        # Streamlit UI setup
        st.set_page_config(page_title="YouTube Video Link Summarizer", page_icon="🎥", layout="wide")
        st.title("🎬 YouTube Video Summarizer")

        # Input the YouTube URL
        url = st.text_input("Enter YouTube Video URL:", key="video_url")

        # User interaction for generating summary
        if st.button("🔍 Generate Summary"):
            if url:
                if self.youtube_summary_generator.is_valid_youtube_url(url):
                    with st.spinner("Generating summary... 🔄"):
                        summary, language = self.youtube_summary_generator.generate_summary(url)
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
                    audio_fp = self.youtube_summary_generator.speak_summary(st.session_state.summary, st.session_state.language)
                    if audio_fp:
                        st.audio(audio_fp, format="audio/mp3")
                    else:
                        st.error("❗ Error generating audio.")
                else:
                    st.warning("❗ Please generate a summary first.")
            else:
                st.warning("❗ Please provide a valid YouTube video URL.")



if __name__ == "__main__":

    # Streamlit secrets for Streamlit Cloud
    groq_api_key = st.secrets["GROQ_API_KEY"]["value"]  
    langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]["value"]
    
    # Initialize the YouTubeSummaryGenerator class with the Groq API key
    youtube_summary_generator = YouTubeSummaryGenerator(groq_api_key)

    # Create an instance of the YouTubeApp class
    youtube_app = YouTubeApp(youtube_summary_generator)

    # Run the app
    youtube_app.run()
