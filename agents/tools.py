import os
import trafilatura
from ddgs import DDGS
import yt_dlp
import traceback
import torch
from transformers import pipeline
from transformers import WhisperForConditionalGeneration, AutoProcessor
import librosa # Although imported, librosa isn't explicitly used for audio loading here. pipeline handles it.
import arxiv
from datetime import datetime # Import datetime module
import re # Import regex for URL validation

# --- Module-level caches for models/pipelines ---
_transcriber_pipeline = None 
_whisper_components = None 

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
        "engadget.com", "zdnet.com", "cnet.com",
        # --- Added more sources ---
        "gizmodo.com", "mashable.com", "digitaltrends.com"
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
        "nytimes.com", "theguardian.com", "cnn.com", "washingtonpost.com",
        # --- Added more sources ---
        "npr.org", "aljazeera.com", "wsj.com"
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
        "cnbc.com", "marketwatch.com", "investopedia.com",
        # --- Added more sources ---
        "reuters.com/business", "seekingalpha.com", "fool.com"
    ]
    site_query_parts = [f"site:{site}" for site in finance_sites]
    full_query = f"{query} ({' OR '.join(site_query_parts)})"
    return search_general_web(full_query, num_results)

def search_youtube_transcripts(query: str, num_videos: int = 1) -> str:
    """
    Searches YouTube for a video on a topic and returns its transcript.
    Uses the model's `generate` method for robust long-form transcription.
    """
    global _whisper_components # Declare intent to use/modify the global cache variable

    print(f" -> Tool: search_youtube_transcripts | Query: '{query}'")
    
    video_url = None
    audio_filename = 'temp_audio'
    audio_path = f'{audio_filename}.mp3'

    try:
        # Step 1: Find and download the YouTube video audio (no changes here)
        youtube_query = f"{query} site:youtube.com"
        print(f"   - Refined search query: '{youtube_query}'")

        with DDGS() as ddgs:
            results = list(ddgs.videos(query=youtube_query, max_results=num_videos, timelimit='m'))
            if not results:
                print("   - No YouTube videos found.")
                return "No YouTube videos found for this topic."
            
            video_url = results[0]['content']
            if not video_url or not re.search(r"youtube\.com|youtu\.be", video_url):
                print(f"   - Search returned a non-YouTube link despite filtering: {video_url}. Skipping.")
                return f"Failed to find a valid YouTube URL for query: {query}"
                
            print(f"   - Found video: '{results[0].get('title', 'Untitled')}' at {video_url}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            'outtmpl': audio_filename, 'quiet': True, 'no_warnings': True, 'restrictfilenames': True,
        }
        print("   - Downloading audio...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print("   - Audio downloaded.")

        # Step 2: 💡 REFACTORED MODEL LOADING
        # Lazy load the model and processor directly instead of the pipeline
        if _whisper_components is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model_id = "distil-whisper/distil-large-v2"
            
            print(f"   - Loading model and processor: '{model_id}' on {device}...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
            model.to(device)
            _whisper_components = {"processor": processor, "model": model, "device": device}
            print("   ✅ Model and processor loaded.")

        # Step 3: 💡 REFACTORED TRANSCRIPTION PROCESS
        # Use the model's native .generate() method for long-form transcription
        print("   - Transcribing audio using model.generate()...")
        
        # Get components from cache
        processor = _whisper_components["processor"]
        model = _whisper_components["model"]
        device = _whisper_components["device"]

        # Load audio file and resample to the required 16kHz
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)

        # Process the audio array to create input features
        input_features = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
        
        # Move features to the correct device (GPU/CPU)
        input_features = input_features.to(device, dtype=torch_dtype if device == "cuda:0" else torch.float32)

        # Generate token IDs using the model's internal chunking
        predicted_ids = model.generate(input_features)

        # Decode the token IDs to text
        transcript_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        print("   - Transcription complete.")
        return transcript_text
        
    except Exception as e:
        print(f"   - Failed during YouTube video processing: {e}")
        traceback.print_exc()
        return f"Failed to get YouTube transcript for '{query}': {e}"
    finally:
        # Step 4: Clean up the audio file (no changes here)
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print("   - Cleaned up temporary audio file.")
            except Exception as e:
                print(f"   - Error cleaning up temp audio file: {e}")


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

    # # Test search_youtube_transcripts
    # print("\n\n--- Testing Functional Tool: search_youtube_transcripts ---")
    # # Use a well-known, public video for testing if possible, or a general topic
    # transcript_results = search_youtube_transcripts("Generative AI 2025")
    # print("\n[RESULTS (YouTube Transcript)]")
    # print(transcript_results[:500] + "...")

    # Test search_arxiv
    print("\n\n--- Testing Functional Tool: search_arxiv ---")
    arxiv_results = search_arxiv("large language model reasoning capabilities")
    print("\n[RESULTS (ArXiv)]")
    print(arxiv_results[:500] + "...")

    # Test get_current_time
    print("\n\n--- Testing New Tool: get_current_time ---")
    current_time = get_current_time()
    print(f"\n[RESULTS]\nCurrent Time: {current_time}")
