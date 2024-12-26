
# YouTube Video Summarizer & Speaker Tool

## Overview

The **YouTube Video Summarizer & Speaker Tool** allows you to input a YouTube video URL, generate a summary of the video's content, and convert that summary into speech. This tool leverages advanced language models for summarization and uses **Google Text-to-Speech (gTTS)** for audio output. 

## Features

- **Summarization**: Automatically summarize YouTube videos based on their transcript.
- **Language Support**: Select from available transcript languages to ensure summaries are in the same language as the video.
- **Speech Output**: Convert the generated summary into speech and play it back.

### Install Dependencies

You can install the required dependencies using `pip` by running:

```bash
pip install -r requirements.txt

### Setup Environment Variables

To run the application, you will need to create a `.env` file in the root directory of the project. This file stores sensitive information, like API keys, in a secure way. Here's how you can set it up:

### Steps to Create the `.env` File:

1. **Create a `.env` file**: In the root directory of the project, create a new file named `.env`.

2. **Add Your API Keys**: Open the `.env` file and add the following environment variables:

   ```env
   LANGCHAIN_API_KEY=<Your_LangChain_API_Key>
   LANGCHAIN_PROJECT=<Your_LangChain_Project_Name>
   GROQ_API_KEY=<Your_GROQ_API_Key>
   HF_TOKEN=<Your_HuggingFace_Token>  # Optional: Only needed for accessing HuggingFace models
