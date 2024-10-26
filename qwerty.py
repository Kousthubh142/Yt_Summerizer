# Simplified imports - use these instead
import os
import shutil
import torch
from transformers import pipeline
import whisper
import librosa
import soundfile as sf
import yt_dlp
import time

# Initialize Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def youtube_to_mp3(youtube_url: str, output_dir: str = "downloads") -> str:
    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "320",
        }],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "verbose": False,  # Changed to False to reduce output noise
    }
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Downloading audio from {youtube_url}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_config) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            # Get the filename directly from ydl info
            filename = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"
            return filename
    except Exception as e:
        print(f"Download error: {e}")
        return ""

def chunk_audio(filename: str, segment_length: int, output_dir: str) -> list:
    print(f"Chunking audio into {segment_length} second segments...")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Use a lower sample rate to reduce memory usage
    audio, sr = librosa.load(filename, sr=16000)
    
    duration = librosa.get_duration(y=audio, sr=sr)
    num_segments = int(duration / segment_length)
    if duration % segment_length != 0:
        num_segments += 1
    
    chunked_files = []
    for i in range(num_segments):
        start = i * segment_length * sr
        end = min((i + 1) * segment_length * sr, len(audio))
        segment = audio[start:end]
        output_file = os.path.join(output_dir, f"segment_{i:03d}.mp3")
        sf.write(output_file, segment, sr)
        chunked_files.append(output_file)
    
    return chunked_files

def transcribe_audio(audio_files: list, output_dir: str = None, model_name="base") -> list:
    print("Transcribing audio...")
    
    model = whisper.load_model(model_name)
    transcripts = []
    
    # Create a transcripts directory if output_dir is provided
    transcripts_dir = os.path.join(output_dir, "transcripts") if output_dir else None
    if transcripts_dir:
        os.makedirs(transcripts_dir, exist_ok=True)
        
    # Create a main transcript file
    main_transcript_file = os.path.join(output_dir, "full_transcript.txt") if output_dir else None
    
    for ind, audio_file in enumerate(audio_files):
        try:
            print(f"Transcribing segment {ind + 1}/{len(audio_files)}")
            result = model.transcribe(audio_file)
            transcript_text = result["text"]
            transcripts.append(transcript_text)
            
            # Save individual segment transcript
            if transcripts_dir:
                segment_file = os.path.join(transcripts_dir, f"segment_{ind+1:03d}_transcript.txt")
                with open(segment_file, "w", encoding="utf-8") as f:
                    f.write(f"Segment {ind+1} Transcript:\n")
                    f.write("="* 50 + "\n\n")
                    f.write(transcript_text)
                    f.write("\n\n")
                    
            # Append to main transcript file
            if main_transcript_file:
                with open(main_transcript_file, "a", encoding="utf-8") as f:
                    f.write(f"\nSegment {ind+1}:\n")
                    f.write("="* 50 + "\n")
                    f.write(transcript_text)
                    f.write("\n\n")
                    
        except Exception as e:
            print(f"Error transcribing segment {ind + 1}: {e}")
            transcripts.append("")
            
            # Log error in files if output_dir is provided
            if transcripts_dir:
                error_file = os.path.join(transcripts_dir, f"segment_{ind+1:03d}_error.txt")
                with open(error_file, "w", encoding="utf-8") as f:
                    f.write(f"Error transcribing segment {ind+1}: {str(e)}")
    
    return transcripts


def init_summarizer():
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")  # Debug output to check device usage
    
    try:
        # Attempt to load the larger model
        return pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            framework="pt",
            device=device  # Pass the device here
        )
    except Exception as e:
        print(f"Error initializing summarizer with large model: {e}")
        # Fallback to a smaller model if the large one fails
        return pipeline(
            "summarization",
            model="facebook/bart-base",
            framework="pt",
            device=device  # Ensure the same device is used
        )


# Modified summarize function with simpler tokenizer handling
def summarize(chunks: list[str], max_length: int = 150, min_length: int = 30) -> list:
    print("Generating summary...")
    
    # Initialize summarizer
    summarizer = init_summarizer()
    summaries = []
    
    for ind, chunk in enumerate(chunks):
        if not chunk.strip():  # Skip empty chunks
            continue
            
        try:
            print(f"Summarizing chunk {ind + 1}/{len(chunks)}")
            
            # Simply truncate the chunk if it's too long
            if len(chunk) > 1024:
                chunk = chunk[:1024]
            
            summary = summarizer(chunk, 
                               max_length=max_length, 
                               min_length=min_length, 
                               do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk {ind + 1}: {e}")
            summaries.append(f"[Summary failed for chunk {ind + 1}]")
    
    return summaries

def summarize_youtube_video(youtube_url: str, output_dir: str) -> tuple:
    try:
        # Create directory structure
        os.makedirs(output_dir, exist_ok=True)
        raw_audio_dir = os.path.join(output_dir, "raw_audio")
        chunks_dir = os.path.join(output_dir, "chunks")
        os.makedirs(raw_audio_dir, exist_ok=True)
        os.makedirs(chunks_dir, exist_ok=True)

        # Clear the full transcript file if it exists
        full_transcript_file = os.path.join(output_dir, "full_transcript.txt")
        if os.path.exists(full_transcript_file):
            os.remove(full_transcript_file)

        # Download audio
        print("Downloading audio...")
        audio_filename = youtube_to_mp3(youtube_url, raw_audio_dir)
        if not audio_filename:
            raise Exception("Failed to download audio")


        # Chunk audio (10-minute segments)
        print("Chunking audio...")
        segment_length = 10 * 60
        chunked_files = chunk_audio(audio_filename, segment_length, chunks_dir)
        
        # Transcribe with output directory
        print("Transcribing audio...")
        transcripts = transcribe_audio(chunked_files, output_dir)
        
        # Generate summaries
        print("Generating detailed summary...")
        long_summaries = summarize(transcripts)
        long_summary = "\n\n".join(long_summaries)
        
        # Generate final short summary
        print("Generating TL;DR summary...")
        short_summary = summarize([long_summary], max_length=32, min_length=30)[0]
        
        # Save summaries
        summary_file = os.path.join(output_dir, "summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Detailed Summary:\n")
            f.write("="* 50 + "\n\n")
            f.write(long_summary)
            f.write("\n\nTL;DR:\n")
            f.write("="* 50 + "\n\n")
            f.write(short_summary)
        
        # print(f"\nFiles saved in {output_dir}:")
        # print(f"1. Full transcript: {os.path.join(output_dir, 'full_transcript.txt')}")
        # print(f"2. Individual transcripts: {os.path.join(output_dir, 'transcripts')} directory")
        # print(f"3. Summary: {summary_file}")
        # print(f"4. Metadata: {metadata_file}")
        
        return long_summary, short_summary

    except Exception as e:
        print(f"Error in summarization process: {e}")
        return None, None

def main():
    print("="*100)
    youtube_url = input("Enter YouTube URL: ")
    outputs_dir = os.path.join(os.getcwd(), "youtube_summaries", 
                              f"summary_{time.strftime('%Y%m%d_%H%M%S')}")
    
    print("\nStarting video summarization process...")
    print(f"Output directory: {outputs_dir}")
    
    long_summary, short_summary = summarize_youtube_video(youtube_url, outputs_dir)
    
    if long_summary and short_summary:
        print("\nProcessing completed successfully!")
        print("\nSummary:")
        print("=" * 100)
        print(short_summary)
    else:
        print("\nFailed to generate summary.")

if __name__ == "__main__":
    main()