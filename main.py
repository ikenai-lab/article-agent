import argparse
import os
import sys
import traceback
import concurrent.futures
import json
import markdown2 # For converting markdown to HTML
from typing import Optional
from dataclasses import asdict # To convert dataclasses to dicts for JSON serialization
import re

# Add the project root directory to the Python path
# This ensures that imports like 'agents.planner' work correctly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import agents and tools
from agents.planner import PlannerAgent, ResearchPlan, PlanStep # Import PlanStep as well
from agents.writer import WriterAgent, RAGContext, ArticlePlan, WritingStyle
# Import AVAILABLE_TOOLS from your tools module
# IMPORTANT: Ensure agents.tools exists and AVAILABLE_TOOLS is a dict of callable functions/objects
try:
    from agents.tools import AVAILABLE_TOOLS
    # Also import build_vector_store if it's in agents.rag_builder
    from agents.rag_builder import build_vector_store 
except ImportError as e:
    print(f"❌ Error: Could not import necessary modules. Details: {e}")
    print("Please ensure 'agents/tools.py' exists and defines 'AVAILABLE_TOOLS',")
    print("and 'agents/rag_builder.py' exists and defines 'build_vector_store'.")
    sys.exit(1)

def execute_research_step(step: PlanStep, available_tools: dict) -> str:
    """
    Executes a single research step using the specified tool and query.
    This is a simplified execution. In a real system, this would involve
    saving data to a raw data directory, handling tool-specific outputs,
    and more sophisticated error handling/retries at the tool level.
    The tool function is expected to return the collected content (string).
    """
    tool_name = step.tool
    query = step.query
    # expected_output = step.expected_output # Not directly used for return, but good for logging

    if tool_name not in available_tools:
        print(f"  ⚠️ Warning: Tool '{tool_name}' not found. Skipping step for query: '{query}'")
        return f"Tool '{tool_name}' not available or failed to execute."

    tool_function = available_tools[tool_name]
    print(f"  Executing tool '{tool_name}' with query: '{query}'...")
    
    try:
        # Call the tool function. It should return the content.
        content = tool_function(query) 
        if content:
            print(f"  ✅ Tool '{tool_name}' completed successfully. Content snippet: {str(content)[:100]}...")
            return content
        else:
            print(f"  ⚠️ Tool '{tool_name}' returned empty content for query: '{query}'.")
            return ""
    except Exception as e:
        print(f"  ❌ Error executing tool '{tool_name}' for '{query}': {e}")
        traceback.print_exc()
        return f"Error executing {tool_name} for '{query}': {e}"


def run_pipeline(topic: str, style_file: str, target_word_count: int, 
                 writing_style: WritingStyle, target_audience: str, force_refresh: bool = False):
    """
    Runs the full agentic pipeline from planning to article generation.
    """
    print("🚀🚀🚀 Starting Agentic Article Generation Pipeline 🚀🚀🚀")
    
    DATA_DIR = 'data'
    # This file will store the combined raw text content from all research tools
    COMBINED_CONTEXT_FILE = os.path.join(DATA_DIR, 'combined_context.txt')
    VECTOR_STORE_DIR = 'vector_store' # Directory for FAISS index and chunks

    # Ensure necessary directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    try:
        # --- Phase 1: Planning & Agentic Research ---
        print("\n--- Running Phase 1: Planning & Agentic Research ---")
        
        # Check for cached combined context to skip research if not forced refresh
        if not force_refresh and os.path.exists(COMBINED_CONTEXT_FILE) and os.path.exists(os.path.join(VECTOR_STORE_DIR, 'faiss_index.bin')):
            print(f"✅ Found cached context at '{COMBINED_CONTEXT_FILE}' and vector store. Skipping research. Use --force-refresh to override.")
            # If skipping research, we still need a dummy research_plan to get the suggested_outline
            planner = PlannerAgent() # Initialize just to get a placeholder
            research_plan = planner.generate_plan(topic, list(AVAILABLE_TOOLS.keys()), complexity_override=PlanComplexity.SIMPLE)
            if not research_plan: # Fallback if even dummy plan fails
                research_plan = ResearchPlan(topic=topic, reasoning="Skipped research, using dummy plan.", complexity=PlanComplexity.SIMPLE, estimated_time="N/A", steps=[], success_criteria=[], fallback_strategies=[], suggested_outline=[f"Introduction to {topic}", "Main Points", "Conclusion"])

        else:
            planner = PlannerAgent()
            tool_names = list(AVAILABLE_TOOLS.keys())
            research_plan: Optional[ResearchPlan] = planner.generate_plan(topic, tool_names)

            if not research_plan or not research_plan.steps: # Check if research_plan is valid and has steps
                print("❌ Could not generate a valid research plan. Aborting.")
                return

            print("\n--- Executing Research Plan ---")
            planner.print_plan_summary(research_plan) # Print the summary of the generated plan
            
            all_content = []
            # Use ThreadPoolExecutor for parallel execution of research steps
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(research_plan.steps))) as executor:
                # Submit tasks for each step in the research plan
                future_to_step = {
                    executor.submit(execute_research_step, step, AVAILABLE_TOOLS): step
                    for step in research_plan.steps
                }
                
                for future in concurrent.futures.as_completed(future_to_step):
                    step = future_to_step[future]
                    try:
                        content = future.result()
                        if content:
                            all_content.append(f"--- Content from Tool: {step.tool}, Query: {step.query} ---\n\n{content}")
                        else:
                            print(f"⚠️ Tool '{step.tool}' returned no content for query: '{step.query}'.")
                    except Exception as exc:
                        print(f"❌ Tool '{step.tool}' generated an exception: {exc}")
                        traceback.print_exc()
            
            if all_content:
                combined_text = "\n\n".join(all_content)
                with open(COMBINED_CONTEXT_FILE, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                print(f"\n✅ Combined research context saved to '{COMBINED_CONTEXT_FILE}'")
            else:
                print("\n❌ No content was gathered from any tool. Aborting.")
                return

            # --- Phase 1.5: RAG Pipeline Build ---
            print("\n--- Running Phase 1.5: Building RAG Knowledge Base ---")
            # The build_vector_store function should read from COMBINED_CONTEXT_FILE
            # and save the FAISS index and chunks to VECTOR_STORE_DIR.
            try:
                build_vector_store([COMBINED_CONTEXT_FILE], vector_store_dir=VECTOR_STORE_DIR) 
                print(f"✅ RAG knowledge base built successfully in '{VECTOR_STORE_DIR}'.")
            except Exception as e:
                print(f"❌ Failed to build RAG knowledge base: {e}")
                traceback.print_exc()
                return

        # --- Phase 2: Article Writing with Critique Loop ---
        print("\n--- Running Phase 2: Writing Article ---")
        
        # Initialize RAGContext (it will load the newly built or existing vector store)
        rag_context = RAGContext(vector_store_dir=VECTOR_STORE_DIR)
        writer_agent = WriterAgent(rag_context=rag_context)
        
        # Step 3: Generate Article Plan (Outline)
        print("\n--- Generating Article Plan (Outline) ---")
        # Pass the suggested_outline from the research_plan to the writer agent
        article_plan: Optional[ArticlePlan] = writer_agent.generate_enhanced_plan(
            topic=topic,
            target_word_count=target_word_count,
            writing_style=writing_style,
            target_audience=target_audience,
            suggested_outline=research_plan.suggested_outline # NEW: Pass suggested outline
        )
        
        if not article_plan:
            print("\n❌❌❌ Pipeline Failed: Could not generate an article plan. ❌❌❌")
            return

        print("\n--- Outline Approval ---")
        print("The following article plan has been generated:")
        print(f"Topic: {article_plan.topic}")
        print(f"Target Word Count: {article_plan.target_word_count}")
        print(f"Writing Style: {article_plan.writing_style.value}")
        print(f"Target Audience: {article_plan.target_audience}")
        print("\nSections:")
        for i, section in enumerate(article_plan.sections):
            dependencies_str = f" (Depends on: {', '.join(section.dependencies)})" if section.dependencies else []
            print(f"  {i+1}. {section.title} [{section.section_type.value}] - {section.target_word_count} words{dependencies_str}")
            print(f"     Key Points: {', '.join(section.key_points)}")
            print(f"     Research Queries: {', '.join(section.research_queries)}")
        
        approve = input("\nDo you approve this outline? (y/n): ").lower().strip()
        
        if approve != 'y':
            print("\n🛑 Outline rejected by user. Aborting pipeline.")
            return
            
        print("✅ Outline approved. Proceeding with article generation...")
        
        # Step 4: Write Article from the approved plan
        final_article_content, metadata = writer_agent.write_article_from_enhanced_plan(article_plan, style_file)

        if final_article_content:
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True) # Ensure outputs directory exists

            # Sanitize topic for filename
            sanitized_topic = re.sub(r'[^\w\s-]', '', topic).replace(' ', '_').lower()
            md_filename = os.path.join(output_dir, f"{sanitized_topic}_article.md")
            html_filename = os.path.join(output_dir, f"{sanitized_topic}_article.html")

            with open(md_filename, 'w', encoding='utf-8') as f:
                f.write(final_article_content)
            print(f"\n🎉🎉🎉 Pipeline Complete! Article saved to '{md_filename}' 🎉🎉🎉")

            # Convert Markdown to HTML and save
            html_content = markdown2.markdown(final_article_content)
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ Article also saved as HTML to: '{html_filename}'")

            # Print final metrics
            writer_agent.print_article_metrics(metadata)
        else:
            print("\n❌❌❌ Pipeline Failed: Article generation did not produce any output. ❌❌❌")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: A required file was not found: {e}")
        print("Please ensure all necessary data (e.g., RAG vector store) is built or available.")
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌❌❌ An unexpected error occurred during the pipeline execution ❌❌❌")
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="An agentic CLI tool to generate a Medium article.")
    
    parser.add_argument('--topic', type=str, required=True, help="The topic of the article to generate.")
    parser.add_argument('--style-file', type=str, required=True, help="Path to a text file containing a writing style sample.")
    parser.add_argument('--target-word-count', type=int, default=2000, help="Target word count for the generated article.")
    parser.add_argument('--writing-style', type=str, default="JOURNALISTIC", 
                        choices=[style.name for style in WritingStyle],
                        help="Overall writing style for the article (e.g., JOURNALISTIC, ACADEMIC).")
    parser.add_argument('--target-audience', type=str, default="general public",
                        help="Target audience for the article.")
    parser.add_argument('--force-refresh', action='store_true', 
                        help="Force research and RAG build even if cached context exists.")

    args = parser.parse_args()
    
    # Create a dummy style file if it doesn't exist for testing
    if not os.path.exists(args.style_file):
        print(f"⚠️ Style file not found. Creating a dummy file at '{args.style_file}'...")
        with open(args.style_file, 'w', encoding='utf-8') as f:
            f.write("This is a sample of my writing. I like to keep things clear and to the point, using active voice and concise sentences. My tone is generally informative and engaging, aiming to educate the reader without being overly formal.")
    
    # For local testing, ensure a dummy agents/tools.py is available or mocked
    # and a dummy agents/rag_builder.py is available or mocked.
    # If agents.tools.py or agents.rag_builder.py are not fully implemented,
    # you might need to provide mocks here.
    if 'AVAILABLE_TOOLS' not in globals():
        print("💡 Mocking AVAILABLE_TOOLS for local main.py test run.")
        class MockTool:
            def __init__(self, name, description="A mock tool"):
                self.name = name
                self.description = description
            def __call__(self, query):
                print(f"  (Mock) Executing {self.name} for query: {query}")
                # Return some mock content for the pipeline to proceed
                return f"Mock content for query: '{query}'. This is some relevant data collected by {self.name}."

        sys.modules['agents.tools'] = type('module', (object,), {
            'AVAILABLE_TOOLS': {
                "web_search": MockTool("web_search", "Performs a web search."),
                "youtube_collector": MockTool("youtube_collector", "Collects YouTube video information."),
                "academic_search": MockTool("academic_search", "Searches academic databases."),
                "news_search": MockTool("news_search", "Searches recent news."),
            }
        })
        from agents.tools import AVAILABLE_TOOLS # Re-import after mocking

    if 'build_vector_store' not in globals():
        print("💡 Mocking build_vector_store for local main.py test run.")
        def mock_build_vector_store(input_files: list, vector_store_dir: str): # Corrected parameter name
            print(f"  (Mock) Building vector store from {input_files} to {vector_store_dir}. (No actual build)")
            # Create dummy FAISS index and chunks for RAGContext to load
            import faiss
            import pickle
            import numpy as np
            dummy_embeddings = np.random.rand(50, 384).astype('float32')
            dummy_chunks = [f"Dummy chunk {i} from mock build. Content from {f}" for f in input_files for i in range(50//len(input_files))]
            
            dummy_index = faiss.IndexFlatL2(dummy_embeddings.shape[1])
            dummy_index.add(dummy_embeddings)
            faiss.write_index(dummy_index, os.path.join(vector_store_dir, 'faiss_index.bin'))
            
            with open(os.path.join(vector_store_dir, 'chunks.pkl'), 'wb') as f:
                pickle.dump(dummy_chunks, f)
            print("  (Mock) Dummy FAISS index and chunks created.")

        sys.modules['agents.rag_builder'] = type('module', (object,), {
            'build_vector_store': mock_build_vector_store
        })
        from agents.rag_builder import build_vector_store # Re-import after mocking


    # Run the pipeline with parsed arguments
    run_pipeline(
        topic=args.topic, 
        style_file=args.style_file, 
        target_word_count=args.target_word_count,
        writing_style=WritingStyle[args.writing_style.upper()], # Ensure enum is passed
        target_audience=args.target_audience,
        force_refresh=args.force_refresh
    )
