> [!CAUTION]
> The current code is underdevelopment and may not work on all systems. Work is being done to improve this.

## Setup & Installation

Before you start, you'll need to set up your environment.

1. Prerequisites

    [Python 3.10](https://www.python.org/downloads/release/python-3100/)

    [Ollama](https://ollama.com/) installed and running.

    [ffmpeg](https://github.com/BtbN/FFmpeg-Builds/releases) installed (required for YouTube audio extraction).

2. Clone the Repository

```bash
git clone https://github.com/ikenai-lab/article-agent.git
cd article-agent
```

3. Set Up a Virtual Environment
It's always a good idea to work in a virtual environment.


### For Unix/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### For Windows
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

4. Install Dependencies
Install all the required Python packages from requirements.txt.

``` bash
pip install -r requirements.txt
``` 

5. Download the LLM
This project is designed to run with local models via Ollama. You'll need to pull the model specified in the agent scripts (the default is gemma3).

``` bash
ollama pull gemma3
```

## Usage

Everything is run from the command line through main.py.

The script requires a few key arguments to get started:

    --topic: The subject of the article you want to generate.

    --style-file: A .txt file containing a sample of your writing.

    --target-word-count: The desired length of the final article.

    --writing-style: The general style (e.g., BLOG, JOURNALISTIC).

    --force-refresh: (Optional) Use this flag to force the tool to re-run the research phase, even if cached data exists.

### Example Command:
``` bash

python main.py --topic "The Future of Artificial Intelligence in Healthcare" --style-file "style_sample.txt" --target-word-count 500 --writing-style "BLOG" --force-refresh

```
The final article will be saved in the /outputs directory as both a Markdown (.md) and an HTML (.html) file.

## Future Development

This project is a living document of my journey, and there's always more to learn and build. Here are the next steps I'm thinking about:

- Obsidian Plugin Pivot: Rather than a standalone web app, a better idea would be to integrate this tool directly into a writer's existing workflow. An Obsidian plugin makes perfect sense. It aligns with the local-first ethos of the project and would allow users to generate articles directly within their personal knowledge base.

- Flexible Model Choice: Currently, the model name is hard-coded in the agent scripts. A key improvement will be to allow users to select which model to use via the command line or a settings file. This would also open up the possibility of using different models for different tasks (e.g., a larger model for the CritiqueAgent and a faster one for the WriterAgent).

- Automated Source Attribution: The RAG pipeline knows exactly which chunks of text were used to generate each section. The final piece of the puzzle is to trace those chunks back to their original source (e.g., the URL or paper) and append a "Sources" section at the end of the generated article, providing full transparency and verifiability.