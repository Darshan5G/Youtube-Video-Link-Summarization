
# YouTube Video Summarizer & Speaker Tool

## Overview

The **YouTube Video Summarizer & Speaker Tool** allows you to input a YouTube video URL, generate a summary of the video's content, and convert that summary into speech. This tool leverages advanced language models for summarization and uses **Google Text-to-Speech (gTTS)** for audio output. 

The application is built using **Streamlit** for a user-friendly interface and **LangChain** for the summarization pipeline.

## Features

- **Summarization**: Automatically summarize YouTube videos based on their transcript.
- **Language Support**: Select from available transcript languages to ensure summaries are in the same language as the video.
- **Speech Output**: Convert the generated summary into speech and play it back.
- **Easy-to-use Interface**: Input the YouTube URL and interact with a simple UI to generate summaries and audio.

## Requirements

To run the application, you need the following libraries and dependencies:

- **Python 3.x** (preferably Python 3.8+)
- **Streamlit**: For the web interface.
- **LangChain**: For the summarization process.
- **YouTube Transcript API**: For fetching the transcript of the video.
- **gTTS**: For converting the text summary into speech.

### Install Dependencies

You can install the required dependencies using `pip` by running:

```bash
pip install -r requirements.txt
