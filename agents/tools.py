import os
import trafilatura
from ddgs import DDGS
import yt_dlp
import traceback
import torch
from transformers import pipeline
import librosa # Although imported, librosa isn't explicitly used for audio loading here. pipeline handles it.
import arxiv
from datetime import datetime # Import datetime module
import re # Import regex for URL validation

# --- Module-level caches for models/pipelines ---
_transcriber_pipeline = None 

# --- Tool Definition ---

class Tool:
    """A simple wrapper class for tools to hold name and description."""
    def __init__(self, name: str, func, description: str):
        self.name = name
        self.func = func
        self.description = description

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

# Actual tool functions
def search_general_web(query: str, num_results: int = 3) -> str:
    """
    Searches the general web for a query and scrapes the top results.
    """
    print(f"  -> Tool: search_general_web | Query: '{query}'")
    all_content = []
    try:
        with DDGS() as ddgs:
            # Added region='wt-wt' for potentially better global results, can be customized
            results = list(ddgs.text(query, max_results=num_results, region='wt-wt')) 
            if not results:
                print("    - No web results found.")
                return "No web results found."

            for result in results:
                url = result.get('href') # Use .get to safely access
                if url:
                    print(f"    - Downloading content from: {url}")
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        content = trafilatura.extract(downloaded, include_links=False, include_comments=False) # Exclude links/comments for cleaner text
                        if content:
                            all_content.append(content)
                        else:
                            print(f"    - No extractable content from {url}.")
                    else:
                        print(f"    - Failed to fetch URL: {url}.")
                else:
                    print(f"    - No URL found in result: {result}")
    except Exception as e:
        print(f"    - Error during general web search: {e}")
        traceback.print_exc()
        return f"Error during web search: {e}"

    if not all_content:
        return "No usable content found from web search."
    return "\n\n---\n\n".join(all_content)

def search_tech_blogs(query: str, num_results: int = 3) -> str:
    """
    A specialized tool that searches only high-quality tech blogs.
    """
    print(f"  -> Tool: search_tech_blogs | Query: '{query}'")
    tech_sites = [
        "techcrunch.com", "theverge.com", "wired.com", 
        "arstechnica.com", "venturebeat.com", "hackernoon.com", 
        "engadget.com", "zdnet.com", "cnet.com" # Added more sites
    ]
    # Corrected site query syntax for DDGS (using site: operator per site)
    site_query_parts = [f"site:{site}" for site in tech_sites]
    # Combine query with site filters using OR for broader reach
    full_query = f"{query} ({' OR '.join(site_query_parts)})"
    return search_general_web(full_query, num_results)

def search_news(query: str, num_results: int = 3) -> str:
    """
    A specialized tool that searches reputable news outlets.
    """
    print(f"  -> Tool: search_news | Query: '{query}'")
    news_sites = [
        "reuters.com", "apnews.com", "bbc.com/news", 
        "nytimes.com", "theguardian.com", "cnn.com", "washingtonpost.com"
    ]
    site_query_parts = [f"site:{site}" for site in news_sites]
    full_query = f"{query} ({' OR '.join(site_query_parts)})"
    return search_general_web(full_query, num_results)

def search_finance_news(query: str, num_results: int = 3) -> str:
    """
    A specialized tool that searches reputable financial news outlets.
    """
    print(f"  -> Tool: search_finance_news | Query: '{query}'")
    finance_sites = [
        "bloomberg.com", "wsj.com", "ft.com", 
        "cnbc.com", "marketwatch.com", "investopedia.com"
    ]
    site_query_parts = [f"site:{site}" for site in finance_sites]
    full_query = f"{query} ({' OR '.join(site_query_parts)})"
    return search_general_web(full_query, num_results)

def search_youtube_transcripts(query: str, num_videos: int = 1) -> str:
    """
    Searches YouTube for a video on a topic and returns its transcript.
    """
    global _transcriber_pipeline # Declare intent to use/modify the global cache variable

    print(f"  -> Tool: search_youtube_transcripts | Query: '{query}'")
    
    video_url = None
    # FIX: Removed the incorrect 'site:youtube.com'
    # DDGS search should naturally return youtube.com links.
    Youtube_query = f"{query} youtube" 
    
    try:
        with DDGS() as ddgs:
            # Use 'videos' search type for more relevant results
            results = list(ddgs.videos(query=Youtube_query, max_results=num_videos))
            if not results:
                print("    - No YouTube videos found.")
                return "No YouTube videos found for this topic."
            
            # Select the first video URL
            video_url = results[0]['content'] # 'content' key typically holds the URL for video results
            
            # Basic validation to ensure it's a YouTube URL
            if not video_url or not re.search(r"youtube\.com|youtu\.be", video_url):
                print(f"    - Found URL is not a valid YouTube link: {video_url}. Skipping download.")
                return f"Invalid YouTube URL found for query: {query}"
                
            print(f"    - Found video: '{results[0].get('title', 'Untitled')}' at {video_url}")

        # Lazy load transcriber pipeline once
        if _transcriber_pipeline is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8 * 1024**3 else torch.float32 # Check for sufficient VRAM
            model_id = "distil-whisper/distil-large-v2"
            print(f"    - Loading transcription model: '{model_id}' on {device}...")
            _transcriber_pipeline = pipeline("automatic-speech-recognition", model=model_id, torch_dtype=torch_dtype, device=device)
            print("    ✅ Transcription model loaded.")

        # Download the audio from the video
        audio_path = 'temp_audio.mp3' # Use a consistent filename
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            'outtmpl': 'temp_audio', # This means the file will be temp_audio.mp3
            'quiet': True,
            'no_warnings': True,
            'restrictfilenames': True, # Keep filenames simple
        }

        print("    - Downloading audio...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # yt_dlp.download returns 0 on success, non-zero on failure
            download_result = ydl.download([video_url])
            if download_result != 0:
                raise Exception("yt-dlp download failed with non-zero exit code.")
        print("    - Audio downloaded.")

        # Transcribe the audio
        print("    - Transcribing audio...")
        # pipeline expects audio path or array. If audio_path is 'temp_audio.mp3', pass that.
        result = _transcriber_pipeline(audio_path, return_timestamps=False) # return_timestamps=False to get just text
        transcript_text = result["text"].strip()
        print("    - Transcription complete.")
        return transcript_text
        
    except Exception as e:
        print(f"    - Failed during YouTube video processing: {e}")
        traceback.print_exc()
        return f"Failed to get YouTube transcript for '{query}': {e}"
    finally:
        # Clean up the downloaded audio file
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print("    - Cleaned up temporary audio file.")
            except Exception as e:
                print(f"    - Error cleaning up temp audio file: {e}")


def search_arxiv(query: str, num_results: int = 2) -> str:
    """
    Searches ArXiv for academic papers and returns their abstracts.
    """
    print(f"  -> Tool: search_arxiv | Query: '{query}'")
    try:
        search = arxiv.Search(
            query=query,
            max_results=num_results,
            sort_by=arxiv.SortCriterion.Relevance,
            # Further refine with filters if needed, e.g., categories
        )
        results = list(search.results())
        if not results:
            print("    - No ArXiv papers found.")
            return "No ArXiv papers found for this topic."
            
        abstracts = [
            f"Title: {result.title}\nAuthors: {', '.join([a.name for a in result.authors])}\nPublished: {result.published.strftime('%Y-%m-%d')}\nAbstract: {result.summary}" 
            for result in results
        ]
        return "\n\n---\n\n".join(abstracts)
    except Exception as e:
        print(f"    - Failed during ArXiv search: {e}")
        traceback.print_exc()
        return f"Failed to search ArXiv: {e}"

def get_current_time() -> str:
    """
    Returns the current date and time in IST.
    """
    print("  -> Tool: get_current_time")
    # Current time is Monday, July 28, 2025 at 8:16:07 PM IST.
    # The datetime.now() function gives local time. No specific timezone conversion needed if local time is IST.
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S IST") # Explicitly add IST

# --- Toolkit for the Planner ---
# Each tool is now wrapped in the Tool class to provide a name and description
AVAILABLE_TOOLS = {
    "search_general_web": Tool("search_general_web", search_general_web, "Search the general web for broad information."),
    "search_tech_blogs": Tool("search_tech_blogs", search_tech_blogs, "Search curated high-quality tech blogs for industry insights."),
    "search_news": Tool("search_news", search_news, "Search reputable news outlets for current events and reports."),
    "search_finance_news": Tool("search_finance_news", search_finance_news, "Search reputable financial news outlets for market data and reports."),
    "search_youtube_transcripts": Tool("search_youtube_transcripts", search_youtube_transcripts, "Search YouTube for videos on a topic and return their transcripts."),
    "search_arxiv": Tool("search_arxiv", search_arxiv, "Search arXiv for academic papers and preprints on scientific topics."),
    "get_current_time": Tool("get_current_time", get_current_time, "Returns the current date and time."),
}

if __name__ == '__main__':
    print("--- Testing tools.py functions ---")
    
    # Test search_general_web
    print("\n--- Testing Tool: search_general_web ---")
    general_web_results = search_general_web("latest AI models 2024", num_results=1)
    print("\n[RESULTS (General Web)]")
    print(general_web_results[:500] + "...")

    # Test search_tech_blogs
    print("\n--- Testing Specialized Tool: search_tech_blogs ---")
    tech_results = search_tech_blogs("Qwen3 release")
    print("\n[RESULTS (Tech Blogs)]")
    print(tech_results[:500] + "...")

    # Test search_finance_news
    print("\n\n--- Testing Specialized Tool: search_finance_news ---")
    finance_results = search_finance_news("NVIDIA earnings report")
    print("\n[RESULTS (Finance News)]")
    print(finance_results[:500] + "...")

    # Test search_youtube_transcripts
    print("\n\n--- Testing Functional Tool: search_youtube_transcripts ---")
    # Use a well-known, public video for testing if possible, or a general topic
    transcript_results = search_youtube_transcripts("Generative AI explained")
    print("\n[RESULTS (YouTube Transcript)]")
    print(transcript_results[:500] + "...")

    # Test search_arxiv
    print("\n\n--- Testing Functional Tool: search_arxiv ---")
    arxiv_results = search_arxiv("large language model reasoning capabilities")
    print("\n[RESULTS (ArXiv)]")
    print(arxiv_results[:500] + "...")

    # Test get_current_time
    print("\n\n--- Testing New Tool: get_current_time ---")
    current_time = get_current_time()
    print(f"\n[RESULTS]\nCurrent Time: {current_time}")

