from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.document_loaders import YoutubeLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import load_summarize_chain
from langchain.schema import Document
from gtts import gTTS
import io
from fastapi.responses import StreamingResponse

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize the FastAPI app
app = FastAPI()

# Function to validate the YouTube URL
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

# Function to generate summary from the YouTube video Link
def generate_summary(url):
    select_lang = select_auto_generated_language(url)[:2]
    loader = YoutubeLoader.from_youtube_url(url, language=[select_lang])
    docs = loader.load()
    transcript = docs[0].page_content

    # Split the transcript into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=len(transcript) // 5, chunk_overlap=50)
    text_chunks = text_splitter.split_text(transcript)

    # Convert text chunks into Document objects
    final_documents = [Document(page_content=chunk) for chunk in text_chunks]

    # Initialize the LLM
    llm = ChatGroq(model="llama-3.2-3b-preview", groq_api_key=groq_api_key)

    # Load the summarization chain
    summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)

    # Generate the summary
    summary = summary_chain.run(final_documents)
    return summary.replace("\n\n", " ").strip(), select_lang

# # Function to convert text summary into speech
# def speak_summary(summary, language):
#     myobj = gTTS(text=summary, lang=language, slow=False)
#     audio_fp = io.BytesIO()
#     myobj.write_to_fp(audio_fp)
#     audio_fp.seek(0)
#     return audio_fp

# Pydantic model for input
class VideoURL(BaseModel):
    url: str

# Endpoint to generate a summary
@app.post("/generate_summary/")
def create_summary(video_url: VideoURL):
    url = video_url.url
    if is_valid_youtube_url(url):
        summary, language = generate_summary(url)
        app.state.summary = summary
        app.state.language = language
        return {"summary": summary, "language": language}
    else:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

@app.get("/speak")
async def speak():
    if hasattr(app.state, 'summary') and hasattr(app.state, 'language'):
        try:
            summary = app.state.summary
            language = app.state.language
            myobj = gTTS(text=summary, lang=language, slow=False)        
            audio_fp = io.BytesIO()
            myobj.write_to_fp(audio_fp)
            audio_fp.seek(0)
            return StreamingResponse(audio_fp, media_type="audio/mp3")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating speech: {e}")
    else:
        raise HTTPException(status_code=400, detail="Summary not generated yet.")

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the YouTube Video Link Summarizer API"}


