import argparse
import os
import sys
import traceback
import concurrent.futures
import json
import markdown2 # For converting markdown to HTML
from typing import Optional, Dict, Any
from dataclasses import asdict # To convert dataclasses to dicts for JSON serialization
import re

# Add the project root directory to the Python path
# This ensures that imports like 'agents.planner' work correctly.
# --- NO CHANGE HERE ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import agents and tools
# --- CHANGE 1: Import the new StyleAnalyzerAgent ---
from agents.planner import PlannerAgent, ResearchPlan, PlanStep
from agents.writer import WriterAgent, RAGContext, ArticlePlan, WritingStyle
from agents.style_analyzer import StyleAnalyzerAgent # New import
# Import AVAILABLE_TOOLS from your tools module
try:
    from agents.tools import AVAILABLE_TOOLS
    from agents.rag_builder import build_vector_store 
except ImportError as e:
    print(f"❌ Error: Could not import necessary modules. Details: {e}")
    sys.exit(1)

def execute_research_step(step: PlanStep, available_tools: dict) -> str:
    # --- NO CHANGE IN THIS FUNCTION ---
    tool_name = step.tool
    query = step.query
    if tool_name not in available_tools:
        print(f"  ⚠️ Warning: Tool '{tool_name}' not found. Skipping step for query: '{query}'")
        return f"Tool '{tool_name}' not available or failed to execute."

    tool_function = available_tools[tool_name]
    print(f"  Executing tool '{tool_name}' with query: '{query}'...")
    
    try:
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
    print("🚀🚀🚀 Starting Agentic Article Generation Pipeline 🚀🚀")
    total_prompt_tokens = 0
    total_eval_tokens = 0
    total_tokens = 0
    
    DATA_DIR = 'data'
    COMBINED_CONTEXT_FILE = os.path.join(DATA_DIR, 'combined_context.txt')
    VECTOR_STORE_DIR = 'vector_store'

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    try:
        # --- Phase 1: Planning & Agentic Research (NO CHANGES HERE) ---
        print("\n--- Running Phase 1: Planning & Agentic Research ---")
        
        if not force_refresh and os.path.exists(COMBINED_CONTEXT_FILE) and os.path.exists(os.path.join(VECTOR_STORE_DIR, 'faiss_index.bin')):
            print(f"✅ Found cached context at '{COMBINED_CONTEXT_FILE}' and vector store. Skipping research. Use --force-refresh to override.")
            planner = PlannerAgent()
            # A dummy plan is still needed to get the suggested_outline for the writer
            research_plan, planner_tokens = planner.generate_plan(topic, list(AVAILABLE_TOOLS.keys()))
            if not research_plan:
                research_plan = ResearchPlan(topic=topic, reasoning="Skipped research, using dummy plan.", complexity="simple", estimated_time="N/A", steps=[], success_criteria=[], fallback_strategies=[], suggested_outline=[f"Introduction to {topic}", "Main Points", "Conclusion"])
        else:
            planner = PlannerAgent()
            tool_names = list(AVAILABLE_TOOLS.keys())
            research_plan, planner_tokens = planner.generate_plan(topic, tool_names)
            total_prompt_tokens += planner_tokens.get('prompt_tokens', 0)
            total_eval_tokens += planner_tokens.get('eval_tokens', 0)
            total_tokens += planner_tokens.get('total_tokens', 0)

            if not research_plan or not research_plan.steps:
                print("❌ Could not generate a valid research plan. Aborting.")
                return

            print("\n--- Executing Research Plan ---")
            planner.print_plan_summary(research_plan)
            
            all_content = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(research_plan.steps))) as executor:
                future_to_step = {executor.submit(execute_research_step, step, AVAILABLE_TOOLS): step for step in research_plan.steps}
                for future in concurrent.futures.as_completed(future_to_step):
                    step = future_to_step[future]
                    try:
                        content = future.result()
                        if content:
                            all_content.append(f"--- Content from Tool: {step.tool}, Query: {step.query} ---\n\n{content}")
                    except Exception as exc:
                        print(f"❌ Tool '{step.tool}' generated an exception: {exc}")
            
            if not all_content:
                print("\n❌ No content was gathered from any tool. Aborting.")
                return

            combined_text = "\n\n".join(all_content)
            with open(COMBINED_CONTEXT_FILE, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            print(f"\n✅ Combined research context saved to '{COMBINED_CONTEXT_FILE}'")

            print("\n--- Running Phase 1.5: Building RAG Knowledge Base ---")
            build_vector_store([COMBINED_CONTEXT_FILE], vector_store_dir=VECTOR_STORE_DIR) 
            print(f"✅ RAG knowledge base built successfully in '{VECTOR_STORE_DIR}'.")

        # --- CHANGE 2: ADDED Style Analysis Phase ---
        print("\n--- Running Phase 1.7: Analyzing Writing Style ---")
        try:
            with open(style_file, 'r', encoding='utf-8') as f:
                style_sample_text = f.read()
            
            style_analyzer = StyleAnalyzerAgent() # Pass your ollama instance here
            style_profile, style_tokens = style_analyzer.analyze_style(style_sample_text)

            total_prompt_tokens += style_tokens.get('prompt_tokens', 0)
            total_eval_tokens += style_tokens.get('eval_tokens', 0)
            total_tokens += style_tokens.get('total_tokens', 0)

            if not style_profile:
                print("❌ Could not generate a style profile. Aborting.")
                return
        except FileNotFoundError:
            print(f"❌ Style file not found at '{style_file}'. Aborting.")
            return

        # --- Phase 2: Article Writing with Critique Loop ---
        print("\n--- Running Phase 2: Writing Article ---")
        
        rag_context = RAGContext(vector_store_dir=VECTOR_STORE_DIR)
        writer_agent = WriterAgent(rag_context=rag_context)
        
        print("\n--- Generating Article Plan (Outline) ---")
        article_plan: Optional[ArticlePlan] = writer_agent.generate_enhanced_plan(
            topic=topic,
            target_word_count=target_word_count,
            writing_style=writing_style,
            target_audience=target_audience,
            suggested_outline=research_plan.suggested_outline
        )
        
        if not article_plan:
            print("\n❌❌❌ Pipeline Failed: Could not generate an article plan. ❌❌❌")
            return

        print("\n--- Outline Approval ---")
        print(f"Topic: {article_plan.topic}")
        print("\nSections:")
        for i, section in enumerate(article_plan.sections):
            dependencies_str = f" (Depends on: {', '.join(section.dependencies)})" if section.dependencies else ""
            print(f"  {i+1}. {section.title} [{section.section_type.value}]{dependencies_str}")
        
        approve = input("\nDo you approve this outline? (y/n): ").lower().strip()
        
        if approve != 'y':
            print("\n🛑 Outline rejected by user. Aborting pipeline.")
            return
            
        print("✅ Outline approved. Proceeding with article generation...")
        
        # --- CHANGE 3: Updated the call to the writer agent ---
        # Pass the generated 'style_profile' dictionary instead of the 'style_file' path.
        final_article_content, metadata, writer_tokens = writer_agent.write_article_from_enhanced_plan(article_plan, style_profile)

        total_prompt_tokens += writer_tokens.get('prompt_tokens', 0)
        total_eval_tokens += writer_tokens.get('eval_tokens', 0)
        total_tokens += writer_tokens.get('total_tokens', 0)

        if final_article_content:
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)

            sanitized_topic = re.sub(r'[^\w\s-]', '', topic).replace(' ', '_').lower()
            md_filename = os.path.join(output_dir, f"{sanitized_topic}_article.md")
            html_filename = os.path.join(output_dir, f"{sanitized_topic}_article.html")

            with open(md_filename, 'w', encoding='utf-8') as f:
                f.write(final_article_content)
            print(f"\n🎉🎉🎉 Pipeline Complete! Article saved to '{md_filename}' 🎉🎉🎉")

            html_content = markdown2.markdown(final_article_content)
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ Article also saved as HTML to: '{html_filename}'")

            writer_agent.print_article_metrics(metadata)
            print("\n--- 📊 Total Token Usage ---")
            print(f"  Total Prompt Tokens: {total_prompt_tokens}")
            print(f"  Total Evaluation Tokens: {total_eval_tokens}")
            print(f"  Grand Total Tokens: {total_tokens}")
            print("--------------------------")
        else:
            print("\n❌❌❌ Pipeline Failed: Article generation did not produce any output. ❌❌❌")

    except Exception as e:
        print(f"\n❌❌❌ An unexpected error occurred during the pipeline execution ❌❌❌")
        traceback.print_exc()

if __name__ == '__main__':
    # --- NO CHANGES IN ARGPARSER ---
    parser = argparse.ArgumentParser(description="An agentic CLI tool to generate a Medium article.")
    
    parser.add_argument('--topic', type=str, required=True, help="The topic of the article to generate.")
    parser.add_argument('--style-file', type=str, required=True, help="Path to a text file containing a writing style sample.")
    parser.add_argument('--target-word-count', type=int, default=1000, help="Target word count for the generated article.")
    parser.add_argument('--writing-style', type=str, default="JOURNALISTIC", 
                        choices=[style.name for style in WritingStyle],
                        help="Overall writing style for the article (e.g., JOURNALISTIC, ACADEMIC).")
    parser.add_argument('--target-audience', type=str, default="general public",
                        help="Target audience for the article.")
    parser.add_argument('--force-refresh', action='store_true', 
                        help="Force research and RAG build even if cached context exists.")

    args = parser.parse_args()
    
    if not os.path.exists(args.style_file):
        print(f"⚠️ Style file not found. Creating a dummy file at '{args.style_file}'...")
        with open(args.style_file, 'w', encoding='utf-8') as f:
            f.write("This is a sample of my writing. I like to keep things clear and to the point, using active voice and concise sentences. My tone is generally informative and engaging, aiming to educate the reader without being overly formal.")
    
    run_pipeline(
        topic=args.topic, 
        style_file=args.style_file, 
        target_word_count=args.target_word_count,
        writing_style=WritingStyle[args.writing_style.upper()],
        target_audience=args.target_audience,
        force_refresh=args.force_refresh
    )
