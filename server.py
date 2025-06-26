from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional
import os
import tempfile
import logging
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
import docx
from youtube_transcript_api import YouTubeTranscriptApi
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify your frontend URL here for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
async def root():
    """Test endpoint to verify server is running"""
    return {"message": "Server is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"Receiving file upload: {file.filename}")
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"Processing file: {file.filename}")
        
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # Process different file types
        if file_ext == '.pdf':
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load_and_split()
            text = "\n".join([page.page_content for page in pages])
        elif file_ext in ['.docx', '.doc']:
            doc = docx.Document(temp_file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_ext == '.txt':
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Get summarizer
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Generate summary
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        
        # Clean up
        os.unlink(temp_file_path)
        
        return JSONResponse(content={"summary": summary})
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/convert")
async def convert_file(
    file: UploadFile = File(...),
    format: str = Form(...)
):
    logger.info(f"Receiving file conversion request: {file.filename} to {format}")
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"Converting file: {file.filename}")
        # Process the file (you can add your conversion logic here)
        result = f"Successfully converted {file.filename} to {format}"
        
        # Clean up
        os.unlink(temp_file_path)
        
        return JSONResponse(content={"result": result})
    except Exception as e:
        logger.error(f"Error converting file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/youtube")
async def process_youtube(url: str = Form(...)):
    logger.info(f"Receiving YouTube URL: {url}")
    try:
        # Extract video ID from URL
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL format")

        # Get video transcript
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            full_transcript = ' '.join([entry['text'] for entry in transcript_list])
        except Exception as e:
            raise ValueError("Could not fetch video transcript. Please ensure the video has subtitles enabled.")

        # Initialize summarizer
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # Process long transcripts in chunks
        max_chunk_size = 1000
        chunks = []
        current_chunk = ""

        # Split transcript into sentences and chunk them
        sentences = full_transcript.split('. ')
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + '. '
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + '. '
        if current_chunk:
            chunks.append(current_chunk)

        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            if len(chunk.strip()) > 50:  # Only summarize chunks with substantial content
                chunk_summary = summarizer(chunk, 
                                        max_length=150, 
                                        min_length=30, 
                                        do_sample=False)[0]['summary_text']
                summaries.append(chunk_summary)

        # Combine all summaries
        final_summary = ' '.join(summaries)

        # If the combined summary is too long, summarize it again
        if len(final_summary.split()) > 150:
            final_summary = summarizer(final_summary, 
                                    max_length=150, 
                                    min_length=50, 
                                    do_sample=False)[0]['summary_text']

        logger.info("Successfully generated summary")
        return JSONResponse(content={
            "summary": final_summary,
            "original_transcript": full_transcript,  # Include original transcript
            "video_id": video_id
        })

    except Exception as e:
        logger.error(f"Error processing YouTube URL: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu.be\/)([^&\n?]*)',
        r'(?:youtube\.com\/embed\/)([^&\n?]*)',
        r'(?:youtube\.com\/v\/)([^&\n?]*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

if __name__ == "__main__":
    logger.info("Starting server on http://localhost:8080")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}") 