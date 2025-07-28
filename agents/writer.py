import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama
import json
import re
import traceback
import numpy as np
import sys
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from datetime import datetime
import hashlib

# Add the project root directory to the Python path
# A check is added to ensure the path is valid before adding it.
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from agents.critique import CritiqueAgent
except (NameError, ImportError):
    # This will allow the script to run standalone without the critique agent
    # for testing purposes, creating a dummy class if it's not found.
    print("⚠️ Warning: Could not import CritiqueAgent. Using a dummy class.")
    class CritiqueAgent:
        def __init__(self, model_name):
            print(f"Dummy CritiqueAgent initialized for model: {model_name}")
        def critique_section(self, *args, **kwargs):
            print("Dummy CritiqueAgent: Performing dummy critique.")
            # Return a reasonable score to allow the pipeline to proceed
            return {'score': 4.5, 'critique': ["Dummy critique: Content is generally good."]}


class WritingStyle(Enum):
    ACADEMIC = "academic"
    JOURNALISTIC = "journalistic"
    BLOG = "blog"
    TECHNICAL = "technical"
    NARRATIVE = "narrative"
    PERSUASIVE = "persuasive"

class SectionType(Enum):
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    ANALYSIS = "analysis"
    CASE_STUDY = "case_study"
    CONCLUSION = "conclusion"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    # Added a generic 'DEFAULT' type for robustness if LLM gives unparseable types
    DEFAULT = "default" 

@dataclass
class WritingMetrics:
    """Metrics for tracking writing quality and performance."""
    word_count: int = 0
    readability_score: float = 0.0 # Placeholder for future implementation
    coherence_score: float = 0.0 # Placeholder for future implementation
    style_consistency: float = 0.0 # Placeholder for future implementation
    factual_accuracy: float = 0.0 # Placeholder for future implementation
    engagement_score: float = 0.0 # Placeholder for future implementation
    refinement_iterations: int = 0
    processing_time: float = 0.0

@dataclass
class SectionPlan:
    """Detailed plan for a single article section."""
    title: str
    section_type: SectionType
    key_points: List[str]
    target_word_count: int
    tone: str  # Moved here: Non-default arguments must come before default arguments
    dependencies: List[str] = field(default_factory=list) # Use default_factory for mutable defaults
    research_queries: List[str] = field(default_factory=list) # Use default_factory
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class ArticlePlan:
    """Comprehensive article planning structure."""
    topic: str
    target_word_count: int
    writing_style: WritingStyle
    target_audience: str
    key_themes: List[str]
    sections: List[SectionPlan]
    research_requirements: List[str]
    success_criteria: List[str]
    estimated_time: str

@dataclass
class SectionResult:
    """Result of section generation with metadata."""
    heading: str
    content: str
    metrics: WritingMetrics
    sources_used: List[str]
    refinement_history: List[str]
    final_score: float

class RAGContext:
    """Enhanced RAG context with better retrieval and caching."""
    def __init__(self, 
                 vector_store_dir='vector_store', 
                 embed_model_name='mixedbread-ai/mxbai-embed-large-v1',
                 reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print("🧠 Initializing Enhanced RAG Context...")
        self.index_path = os.path.join(vector_store_dir, 'faiss_index.bin')
        self.chunks_path = os.path.join(vector_store_dir, 'chunks.pkl')

        if not os.path.exists(self.index_path) or not os.path.exists(self.chunks_path):
            raise FileNotFoundError(
                f"Vector store not found at {vector_store_dir}. "
                "Please run rag_builder.py first to create 'faiss_index.bin' and 'chunks.pkl'."
            )

        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.chunks)} chunks.")
        except Exception as e:
            raise IOError(f"Failed to load FAISS index or chunks: {e}")

        # Lazy load models or ensure they are loaded only once if RAGContext is a singleton
        # For simplicity, they are loaded here, assuming RAGContext is instantiated once.
        try:
            self.embedding_model = SentenceTransformer(embed_model_name)
            self.reranker = CrossEncoder(reranker_model_name)
            print("Embedding and Reranker models loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding or reranker models: {e}")
        
        # Add query caching to avoid redundant retrievals
        self._query_cache = {}
        self._lock = threading.Lock() # For thread-safe cache access
        
        print("✅ Enhanced RAG Context ready.")

    def query(self, query_text: str, initial_k: int = 15, final_top_n: int = 5, 
              use_cache: bool = True) -> Tuple[List[str], List[float]]:
        """Enhanced query method with caching and confidence scores."""
        query_hash = hashlib.md5(f"{query_text}_{initial_k}_{final_top_n}".encode()).hexdigest()
        
        if use_cache:
            with self._lock: # Ensure thread-safe access to cache
                if query_hash in self._query_cache:
                    print(f"   -> Cache hit for query: '{query_text[:50]}...'")
                    return self._query_cache[query_hash]
        
        # Embed and retrieve
        # Ensure query_embedding is 2D array for FAISS search
        query_embedding = self.embedding_model.encode([query_text], convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        
        # Handle empty index case
        if self.index.ntotal == 0:
            print("   -> FAISS index is empty. No retrieval possible.")
            return [], []

        similarities, indices = self.index.search(query_embedding, initial_k)
        
        # FIX: Revised logic for filtering and selecting chunks/similarities
        # This ensures that the indices we use to access 'self.chunks' are valid
        # AND that the corresponding similarities are kept.
        # We also filter out any FAISS results that are outside the initial_k range
        # which might sometimes occur with certain FAISS configurations or edge cases.
        
        # Combine similarities and indices into (similarity, original_index) pairs
        # Then filter for validity: index not -1 and index within self.chunks bounds
        combined_faiss_results = []
        # Iterate up to initial_k, as similarities[0] only has initial_k elements
        for i in range(min(initial_k, len(indices[0]))): # Ensure we don't go out of bounds for indices[0]
            idx = indices[0][i]
            if idx != -1 and idx < len(self.chunks):
                combined_faiss_results.append((similarities[0][i], idx))
        
        # Sort these combined results by similarity (descending)
        combined_faiss_results.sort(key=lambda x: x[0], reverse=True)
        
        # Select the top 'initial_k' valid results for reranking
        # This ensures we only pass valid chunks and their corresponding similarities to the reranker
        retrieved_chunks = [self.chunks[idx] for sim, idx in combined_faiss_results[:initial_k]]
        base_similarities = [sim for sim, idx in combined_faiss_results[:initial_k]]
        
        if not retrieved_chunks:
            print("   -> No valid chunks retrieved after initial FAISS search or filtering.")
            return [], []

        pairs = [[query_text, chunk] for chunk in retrieved_chunks]
        
        try:
            rerank_scores = self.reranker.predict(pairs)
            # Ensure rerank_scores is 1D and matches base_similarities length
            if rerank_scores.ndim > 1:
                rerank_scores = rerank_scores.flatten() 
            
            # Handle potential length mismatch between base_similarities and rerank_scores
            # This can happen if the reranker filters out some pairs or if there's an internal issue.
            if len(rerank_scores) != len(base_similarities):
                print(f"   CRITICAL RAG ERROR: Reranker scores length mismatch with base similarities.")
                print(f"   Base similarities length: {len(base_similarities)}, Rerank scores length: {len(rerank_scores)}")
                # Fallback: Truncate both to the minimum length to avoid IndexError
                min_len = min(len(base_similarities), len(rerank_scores))
                base_similarities = base_similarities[:min_len]
                rerank_scores = rerank_scores[:min_len]
                print(f"   Adjusted lengths to {min_len} for combination.")

            combined_scores = 0.3 * np.array(base_similarities) + 0.7 * np.array(rerank_scores)
        except Exception as e:
            print(f"   CRITICAL RAG ERROR: Reranker prediction failed for query: '{query_text[:50]}...'. Error: {e}")
            traceback.print_exc()
            # Fallback: If reranker fails, use only base similarities
            print("   Falling back to using only base similarities for scoring.")
            combined_scores = np.array(base_similarities) # Use only base similarities
            
        # Ensure combined_scores is not empty before zipping
        if combined_scores.size == 0:
            print("   -> Combined scores array is empty after reranking/fallback.")
            return [], []

        # Sort by combined score and take top N
        scored_chunks = list(zip(combined_scores.tolist(), retrieved_chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        top_chunks = [chunk for score, chunk in scored_chunks[:final_top_n]]
        top_scores = [float(score) for score, chunk in scored_chunks[:final_top_n]]
        
        result = (top_chunks, top_scores)
        
        # Cache the result
        if use_cache:
            with self._lock:
                self._query_cache[query_hash] = result
                
        return result

    def get_contextual_chunks(self, queries: List[str], max_chunks: int = 8) -> Tuple[List[str], Dict[str, float]]:
        """Retrieve chunks for multiple related queries and deduplicate.
        Uses a more robust deduplication by checking content similarity if needed,
        but for now relies on exact string match which is efficient.
        """
        all_chunks = []
        chunk_scores = {}
        
        # Distribute max_chunks across queries, ensuring at least 1 per query
        chunks_per_query = max(1, max_chunks // len(queries)) if queries else max_chunks
        
        for query in queries:
            chunks, scores = self.query(query, final_top_n=chunks_per_query + 2) # Fetch a few more for better selection
            for chunk, score in zip(chunks, scores):
                # Simple deduplication: if chunk content is already present, keep the higher score
                if chunk not in chunk_scores or score > chunk_scores[chunk]:
                    chunk_scores[chunk] = score
                    if chunk not in all_chunks: # Only add to all_chunks if it's a new unique piece of content
                        all_chunks.append(chunk)
        
        # Sort by score and take top chunks
        sorted_chunks = sorted(all_chunks, key=lambda x: chunk_scores[x], reverse=True)
        return sorted_chunks[:max_chunks], {chunk: chunk_scores[chunk] for chunk in sorted_chunks[:max_chunks]}


class WriterAgent:
    """Enhanced writer agent with sophisticated planning and quality control."""
    
    def __init__(self, rag_context: RAGContext, model_name="gemma3", critique_model_name="gemma3"):
        print("✍️ Initializing Enhanced Writer Agent...")
        self.rag_context = rag_context
        # Pass model_name to CritiqueAgent if it uses an LLM
        self.critique_agent = CritiqueAgent(model_name=critique_model_name) 
        self.model_name = model_name
        
        # Configuration
        self.max_refinements = 3
        self.min_quality_threshold = 3.5 # Adjusted threshold for initial testing
        self.parallel_processing = True # Enable/Disable parallel processing of sections
        
        # State tracking (for potential future use with session management)
        self.current_article_context = {}
        self.writing_history = []
        
        try:
            ollama.show(self.model_name)
            print(f"✅ Ollama model '{self.model_name}' found for WriterAgent.")
        except Exception:
            print(f"❌ Ollama model '{self.model_name}' not found for WriterAgent.")
            print(f"   Please follow the setup instructions to create it.")
            raise

    def _generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Enhanced response generation with configurable parameters."""
        try:
            response = ollama.chat(
                model=self.model_name, 
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error during Ollama generation: {e}")
            return ""

    def _analyze_writing_style(self, style_sample: str) -> Dict[str, str]:
        """Analyze the writing style sample to extract key characteristics."""
        print("     -> Analyzing writing style characteristics...")
        
        # Limit sample to prevent exceeding context window for LLM
        sample_to_analyze = style_sample[:3000] # Use a larger sample if available, or truncate

        analysis_prompt = f"""
Analyze this writing sample and extract its key stylistic characteristics.
Focus on objective descriptors rather than subjective opinions.

WRITING SAMPLE:
---
{sample_to_analyze}
---

Provide a JSON response with these characteristics:
{{
  "tone": "description of the overall tone (e.g., formal, informal, objective, subjective, enthusiastic)",
  "voice": "description of the narrative voice (e.g., first person, third person, authoritative, conversational)",
  "sentence_structure": "typical sentence patterns and complexity (e.g., varied, simple, complex, compound)",
  "vocabulary_level": "academic/professional/casual/technical/accessible",
  "rhetorical_devices": "key rhetorical techniques used (e.g., analogies, metaphors, rhetorical questions, statistics)",
  "paragraph_style": "how paragraphs are structured (e.g., short, long, topic-sentence driven, flowing)",
  "engagement_techniques": "how the writer engages readers (e.g., direct address, storytelling, data presentation, humor)"
}}

Respond with ONLY the JSON object. Do not include any other text, preamble, or markdown code blocks.
"""
        
        response = self._generate_response(analysis_prompt, temperature=0.3)
        try:
            # Use non-greedy regex to find the first JSON object
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON object found in response", response, 0)
            
            json_string = json_match.group(0)
            
            # Robustly remove trailing commas before parsing
            # This regex targets commas followed by whitespace and then a closing bracket or brace.
            json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
            
            style_analysis = json.loads(json_string)
            print("     ✅ Style analysis parsed successfully.")
            return style_analysis
        except (AttributeError, json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Could not parse style analysis ({e}), using default characteristics.")
            traceback.print_exc() # Print traceback for debugging
            return {
                "tone": "informative and objective",
                "voice": "third person, authoritative",
                "sentence_structure": "varied, leaning towards clear and concise",
                "vocabulary_level": "professional and accessible",
                "rhetorical_devices": "factual presentation, logical arguments",
                "paragraph_style": "well-structured, topic-sentence driven",
                "engagement_techniques": "clear explanations, data-driven insights"
            }

    def generate_enhanced_plan(self, topic: str, target_word_count: int = 2000, 
                               writing_style: WritingStyle = WritingStyle.JOURNALISTIC,
                               target_audience: str = "general public",
                               suggested_outline: Optional[List[str]] = None) -> Optional[ArticlePlan]: # NEW: suggested_outline parameter
        """Generate a comprehensive article plan with detailed section planning."""
        print(f"📋 Generating enhanced article plan for: '{topic}'...")
        
        # Get all valid SectionType values for the prompt
        valid_section_types = "|".join([st.value for st in SectionType if st != SectionType.DEFAULT]) # Exclude DEFAULT from prompt

        # NEW: Incorporate suggested_outline into the prompt
        outline_guidance = ""
        if suggested_outline:
            # Add an example to guide the LLM on how to map suggested outline to sections
            outline_list_str = "\n".join([f"- {item}" for item in suggested_outline])
            outline_guidance = f"""
SUGGESTED HIGH-LEVEL OUTLINE (from PlannerAgent):
Use these as a strong guide for your "sections" titles and overall structure.
For example, if the suggested outline has "Introduction: Overview", your section title should be "Introduction: Overview" or a very close variant.
{outline_list_str}

"""

        planning_prompt = f"""
You are an expert content strategist creating a comprehensive article plan.

ARTICLE REQUIREMENTS:
- Topic: "{topic}"
- Target word count: {target_word_count} words
- Writing style: {writing_style.value}
- Target audience: {target_audience}

{outline_guidance}

Create a detailed plan following this EXACT JSON structure. Ensure the final output is a single, valid JSON object with no extraneous text or formatting issues.

{{
  "topic": "{topic}",
  "target_word_count": {target_word_count},
  "writing_style": "{writing_style.value}",
  "target_audience": "{target_audience}",
  "key_themes": ["theme1", "theme2", "theme3"],
  "sections": [
    {{
      "title": "Section Title",
      "section_type": "{valid_section_types}",
      "key_points": ["point1", "point2", "point3"],
      "target_word_count": 400,
      "dependencies": [],
      "research_queries": ["query1", "query2"],
      "tone": "engaging|analytical|informative|persuasive|neutral",
      "priority": 1
    }}
  ],
  "research_requirements": ["specific research needs"],
  "success_criteria": ["criteria for successful article"],
  "estimated_time": "time estimate (e.g., '1-2 hours')"
}}

PLANNING GUIDELINES:
1. Create 5-7 sections with a logical flow.
2. Distribute the total target word count ({target_word_count} words) reasonably across sections. For example, an introduction might be 100-150 words, main sections 200-300 words, and conclusion 100-150 words.
3. Include specific, relevant research queries for each section.
4. Set dependencies where sections build on each other. For dependencies, use the *EXACT, FULL TITLE* of the section it depends on. E.g., if a section is titled "Introduction: The AI Revolution", and another section depends on it, the dependency should be "Introduction: The AI Revolution", not just "Introduction".
5. For "section_type", choose *only one* from the valid types: {valid_section_types}. Do NOT combine them.
6. IMPORTANT: Ensure there are no trailing commas in JSON lists or objects. Every object in a list must be separated by a comma, except the last one.
7. Ensure all string values in the JSON are properly quoted.
8. If a SUGGESTED HIGH-LEVEL OUTLINE is provided, use its titles as the primary basis for your "sections" titles. Expand on them as needed, but maintain their core meaning and order.

Provide ONLY the JSON object. No additional text, markdown code blocks, or conversational filler.
"""
        
        response = self._generate_response(planning_prompt, temperature=0.5)
        
        try:
            # FIX: More robust JSON extraction and repair
            # 1. Try to find the JSON object.
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON object found in response", response, 0)
            
            json_string = json_match.group(0)
            
            # 2. Aggressively clean the JSON string to fix common LLM formatting errors.
            #    - Remove trailing commas before a closing bracket or brace.
            json_string = re.sub(r",\s*([}\]])", r"\1", json_string)
            #    - Add missing commas between objects in a list (e.g., [ {{...}} {{...}} ] -> [ {{...}}, {{...}} ])
            json_string = re.sub(r'\}\s*\{', '}, {', json_string)
            #    - Attempt to close unclosed lists/objects at the end of the string
            #      This is a heuristic and might not always produce valid JSON if truncation is severe.
            open_brackets = json_string.count('{') - json_string.count('}')
            open_square_brackets = json_string.count('[') - json_string.count(']')
            
            if open_brackets > 0:
                json_string += '}' * open_brackets
            if open_square_brackets > 0:
                json_string += ']' * open_square_brackets
            
            # Try to parse the cleaned JSON
            plan_dict = json.loads(json_string)
            
            # Convert to ArticlePlan object and validate enums/types
            sections = []
            # Store actual section titles for dependency validation
            # FIX: Get titles from the *parsed* plan_dict sections, not from suggested_outline
            # This is crucial for validating dependencies against what the LLM *actually* generated.
            parsed_section_titles = {s_data.get('title') for s_data in plan_dict.get('sections', []) if s_data.get('title')}

            for section_data in plan_dict.get('sections', []):
                # FIX: Validate and clean dependencies
                raw_dependencies = section_data.get('dependencies', [])
                cleaned_dependencies = []
                for dep in raw_dependencies:
                    dep_title_str = str(dep).strip()
                    if dep_title_str in parsed_section_titles: # Check against parsed titles
                        cleaned_dependencies.append(dep_title_str)
                    else:
                        print(f"⚠️ Warning: LLM-generated dependency '{dep_title_str}' for section '{section_data.get('title', 'Untitled')}' does not match an existing section title. Removing this dependency.")
                
                # FIX: More robust SectionType parsing
                section_type_str_raw = section_data.get('section_type', 'analysis').upper()
                # Try to find a valid SectionType from the raw string
                found_section_type = SectionType.DEFAULT # Default fallback
                for st_enum in SectionType:
                    if st_enum != SectionType.DEFAULT and st_enum.value.upper() in section_type_str_raw:
                        found_section_type = st_enum
                        break
                section_type = found_section_type
                if section_type == SectionType.DEFAULT:
                       print(f"⚠️ Unknown or unparseable section type '{section_type_str_raw}', using 'DEFAULT'")
                
                section = SectionPlan(
                    title=section_data.get('title', 'Untitled Section'),
                    section_type=section_type,
                    key_points=section_data.get('key_points', []),
                    target_word_count=section_data.get('target_word_count', 300),
                    tone=section_data.get('tone', 'informative'), # Ensure tone is passed
                    dependencies=cleaned_dependencies, # Use cleaned dependencies
                    research_queries=section_data.get('research_queries', []),
                    priority=section_data.get('priority', 1)
                )
                sections.append(section)
            
            # Validate writing_style
            writing_style_str = plan_dict.get('writing_style', writing_style.value).upper() # Convert to uppercase
            try:
                validated_writing_style = WritingStyle[writing_style_str]
            except KeyError:
                print(f"⚠️ Unknown writing style '{writing_style_str}', using default '{writing_style.value}'")
                validated_writing_style = writing_style
            
            article_plan = ArticlePlan(
                topic=plan_dict.get('topic', topic), # Use topic from input if not in plan_dict
                target_word_count=plan_dict.get('target_word_count', target_word_count),
                writing_style=validated_writing_style,
                target_audience=plan_dict.get('target_audience', target_audience),
                key_themes=plan_dict.get('key_themes', []),
                sections=sections,
                research_requirements=plan_dict.get('research_requirements', []),
                success_criteria=plan_dict.get('success_criteria', []),
                estimated_time=plan_dict.get('estimated_time', 'Not specified')
            )
            
            print("✅ Enhanced article plan generated successfully.")
            return article_plan
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"❌ Failed to parse article plan: {e}")
            print(f"LLM Response was: {response[:500]}...") # Print first 500 chars for debugging
            traceback.print_exc() # Print full traceback
            return None

    def _enhance_context_with_dependencies(self, section: SectionPlan, completed_sections: Dict[str, str]) -> str:
        """Enhance context by including relevant content from dependent sections."""
        dependency_context = ""
        
        for dep_title in section.dependencies:
            # Ensure dep_title is a string key for lookup
            dep_title_str = str(dep_title) 
            if dep_title_str in completed_sections:
                # Extract key points from the dependency, or summarize using LLM if content is too long
                dep_content = completed_sections[dep_title_str]
                # Simple truncation for now, can be enhanced with LLM summarization
                truncated_content = dep_content[:500] + "..." if len(dep_content) > 500 else dep_content
                dependency_context += f"\n\n--- Content from '{dep_title_str}' ---\n{truncated_content}\n--- End of '{dep_title_str}' ---\n"
            else:
                print(f"   ⚠️ Warning: Dependent section '{dep_title_str}' not found in completed sections for '{section.title}'.")
        
        return dependency_context

    def _compress_and_synthesize_context(self, chunks: List[str], confidence_scores: Dict[str, float], 
                                         section: SectionPlan, max_synthesis_words: int = 400) -> str:
        """Advanced context compression with synthesis for specific section needs."""
        print("     -> Synthesizing context for section requirements...")
        
        if not chunks:
            return "No research context was found for this section."

        # Weight chunks by confidence and relevance to section key points
        weighted_chunks = []
        for chunk in chunks:
            score = confidence_scores.get(chunk, 0.0) # Get score from dict
            relevance_bonus = 0
            for key_point in section.key_points:
                if any(word.lower() in chunk.lower() for word in key_point.split()):
                    relevance_bonus += 0.1 # Small bonus for keyword match
            
            weighted_score = score + relevance_bonus
            weighted_chunks.append((chunk, weighted_score))
        
        # Sort by weighted score and take top chunks (e.g., top 8 for synthesis, giving LLM more to work with)
        weighted_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks_for_synthesis = [chunk for chunk, _ in weighted_chunks[:8]] # Increased from 6 to 8
        
        chunks_text = "\n\n---CHUNK---\n\n".join(top_chunks_for_synthesis)
        key_points_str = ", ".join(section.key_points)
        
        synthesis_prompt = f"""
Synthesize the following research chunks into a coherent, factual summary specifically focused on these key points: {key_points_str}

RESEARCH CHUNKS:
{chunks_text}

SYNTHESIS REQUIREMENTS:
1. Focus on information relevant to: {key_points_str}
2. Organize facts logically.
3. Preserve specific data, numbers, and quotes where crucial.
4. Remove redundant information.
5. Maintain factual accuracy.
6. Create smooth narrative flow.
7. Target a concise summary, ideally around {max_synthesis_words} words.
8. The synthesis should be comprehensive and provide a strong factual foundation for writing the article section.

CRITICAL: Output only the comprehensive but concise synthesis. Do not include any preamble or explanation.
"""
        
        return self._generate_response(synthesis_prompt, temperature=0.4, max_tokens=int(max_synthesis_words * 1.2)) # Allow some buffer

    def _write_section_with_enhanced_context(self, article_plan: ArticlePlan, section: SectionPlan, 
                                             style_analysis: Dict[str, str], 
                                             completed_sections: Dict[str, str]) -> SectionResult:
        """Write a single section with enhanced context and style adherence."""
        start_time = datetime.now()
        print(f"\n   🎯 Writing section: '{section.title}' ({section.section_type.value})")
        
        # Gather context from multiple sources
        # Combine section-specific queries with a general query for the topic + section title
        all_research_queries = section.research_queries + [f"{article_plan.topic} {section.title}" if section.title else article_plan.topic] # Ensure a valid query
        research_chunks, confidence_scores_dict = self.rag_context.get_contextual_chunks(
            all_research_queries,
            max_chunks=8 # Number of top chunks to retrieve initially
        )

        # Synthesize context
        synthesized_context = self._compress_and_synthesize_context(
            research_chunks, confidence_scores_dict, section # Pass dict for confidence scores
        )
        
        # Add dependency context
        dependency_context = self._enhance_context_with_dependencies(section, completed_sections)
        
        # Build comprehensive writing prompt
        section_prompt = f"""
You are an expert writer crafting a specific section of an article. Follow these requirements precisely:

ARTICLE CONTEXT:
- Overall topic: "{article_plan.topic}"
- Target audience: {article_plan.target_audience}
- Section title: "{section.title}"
- Section type: {section.section_type.value}
- Target word count: {section.target_word_count} words
- Tone for this section: {section.tone}

KEY POINTS TO ADDRESS IN THIS SECTION:
{chr(10).join(f"• {point}" for point in section.key_points)}

WRITING STYLE REQUIREMENTS:
- Tone: {style_analysis['tone']}
- Voice: {style_analysis['voice']}
- Sentence structure: {style_analysis['sentence_structure']}
- Vocabulary level: {style_analysis['vocabulary_level']}
- Engagement techniques: {style_analysis['engagement_techniques']}
- Paragraph style: {style_analysis['paragraph_style']}
- Rhetorical devices: {style_analysis['rhetorical_devices']}

RESEARCH CONTEXT (Synthesized from multiple sources):
---
{synthesized_context}
---

DEPENDENCY CONTEXT (Content from previous, related sections):
---
{dependency_context if dependency_context else "No dependency context available."}
---

WRITING INSTRUCTIONS:
1.  Write ONLY the body content for this section. Do NOT repeat the section title.
2.  **CRITICAL**: Your primary goal is to build upon the content from previous sections provided in the "DEPENDENCY CONTEXT". **Do NOT repeat facts, themes, or examples already covered in it.** Your task is to write the *next* logical part of the article.
3.  Address all key points for *this section* naturally and comprehensively.
4.  Use the "RESEARCH CONTEXT" to introduce new facts and support your new points.
5.  Maintain the specified writing style and tone throughout.
6.  Ensure smooth transitions from the previous section into this one.
7.  Target approximately {section.target_word_count} words and ensure the section ends with a complete sentence.

CRITICAL: Your response must contain ONLY the article section's content. Do not add any conversational text, headings, or explanations.
"""
        
        draft_content = self._generate_response(section_prompt, temperature=0.7, max_tokens=int(section.target_word_count * 2.0))
        
        # POST-PROCESSING FIX: Remove repeated header from the generated content
        # Check if the generated content starts with the section title (case-insensitive, ignoring whitespace)
        if draft_content.strip().lower().startswith(section.title.strip().lower()):
            lines = draft_content.strip().splitlines()
            if lines and lines[0].strip().lower() == section.title.strip().lower():
                print(f"     -> Post-processing: Removed repeated header '{lines[0].strip()}' from content.")
                draft_content = "\n".join(lines[1:]).strip()

        # Initialize metrics
        metrics = WritingMetrics()
        refinement_history = []
        # Store sources used (first 100 chars of top 3 chunks for brevity)
        sources_used = [chunk[:100] + "..." for chunk in research_chunks[:3]]
        
        # Refine the section iteratively
        final_score = 0.0
        for iteration in range(self.max_refinements):
            print(f"     -> Running critique for iteration {iteration + 1}...")
            critique_result = self.critique_agent.critique_section(
                article_plan.topic, section.title, draft_content, 
                synthesized_context, json.dumps(style_analysis) # Pass style_analysis as JSON string
            )
            
            current_score = critique_result.get('score', 0.0) # Ensure score is float
            critiques = critique_result.get('critique', [])
            
            print(f"     📊 Iteration {iteration + 1}: Quality score {current_score:.1f}/{self.min_quality_threshold} (min required)")
            
            if current_score >= self.min_quality_threshold:
                final_score = current_score
                print(f"     ✅ Section meets quality threshold ({current_score:.1f}/5.0).")
                break
            elif critiques and iteration < self.max_refinements - 1:
                print(f"     🔄 Refining section based on feedback: {critiques}")
                refinement_history.append(f"Iteration {iteration + 1}: {critiques}")
                draft_content = self._rewrite_section_with_enhanced_feedback(
                    article_plan, section, draft_content, critiques, style_analysis, synthesized_context
                )
            else:
                final_score = current_score
                print(f"     🛑 Max refinements reached or no critiques to apply. Final score: {current_score:.1f}/5.0.")
                break
        
        # Calculate final metrics for this section
        processing_time = (datetime.now() - start_time).total_seconds()
        metrics.word_count = len(draft_content.split())
        metrics.processing_time = processing_time
        metrics.refinement_iterations = len(refinement_history)
        
        return SectionResult(
            heading=section.title,
            content=draft_content,
            metrics=metrics,
            sources_used=sources_used,
            refinement_history=refinement_history,
            final_score=final_score
        )

    def _rewrite_section_with_enhanced_feedback(self, article_plan: ArticlePlan, section: SectionPlan,
                                                draft_content: str, critiques: List[str], 
                                                style_analysis: Dict[str, str], context: str) -> str:
        """Enhanced section rewriting with detailed feedback incorporation."""
        critique_points = "\n".join(f"• {critique}" for critique in critiques)
        
        rewrite_prompt = f"""
You are revising a section of an article based on editorial feedback. Make targeted improvements while maintaining style consistency.

ORIGINAL ARTICLE TOPIC: "{article_plan.topic}"
SECTION TITLE: "{section.title}"
TARGET WORD COUNT: {section.target_word_count} words
SECTION TYPE: {section.section_type.value}

CURRENT DRAFT:
---
{draft_content}
---

EDITORIAL FEEDBACK TO ADDRESS:
{critique_points}

REQUIRED WRITING STYLE:
- Tone: {style_analysis['tone']}
- Voice: {style_analysis['voice']}
- Sentence structure: {style_analysis['sentence_structure']}
- Vocabulary level: {style_analysis['vocabulary_level']}
- Engagement techniques: {style_analysis['engagement_techniques']}
- Paragraph style: {style_analysis['paragraph_style']}
- Rhetorical devices: {style_analysis['rhetorical_devices']}

SUPPORTING CONTEXT (for factual accuracy):
---
{context[:1000]}... # Limit context to avoid exceeding token window
---

REVISION INSTRUCTIONS:
1. Address each point in the editorial feedback specifically and thoroughly.
2. Maintain the target word count ({section.target_word_count} words) as closely as possible.
3. Preserve the required writing style throughout the revised content.
4. Improve clarity, flow, and engagement as per feedback.
5. Ensure factual accuracy using the provided supporting context.
6. Maintain the section's original focus and purpose.
7. Do NOT repeat the section title or any part of it in the body content.

CRITICAL: Provide ONLY the complete, revised section content. Do not include any explanations, preambles, or conversational text.
"""
        
        return self._generate_response(rewrite_prompt, temperature=0.6, max_tokens=int(section.target_word_count * 2.0))

    def write_article_from_enhanced_plan(self, article_plan: ArticlePlan, style_file: str) -> Tuple[str, Dict[str, any]]:
        """Write a complete article from an enhanced plan with detailed tracking."""
        print(f"\n✍️ Writing article: '{article_plan.topic}'")
        print(f"   📊 Target: {article_plan.target_word_count} words, Style: {article_plan.writing_style.value}")
        
        # Load and analyze writing style
        style_analysis = {}
        try:
            with open(style_file, 'r', encoding='utf-8') as f:
                style_sample = f.read()
            style_analysis = self._analyze_writing_style(style_sample)
        except FileNotFoundError:
            print(f"❌ Style file not found: {style_file}. Using default style analysis.")
            style_analysis = self._analyze_writing_style("") # Get default analysis
        except Exception as e:
            print(f"❌ Error loading/analyzing style file: {e}. Using default style analysis.")
            traceback.print_exc()
            style_analysis = self._analyze_writing_style("")

        # Sort sections by dependencies to establish writing order
        sorted_sections = self._sort_sections_by_dependencies(article_plan.sections)
        print(f"  Order of sections after dependency sorting: {[s.title for s in sorted_sections]}")
        
        # Initialize containers for results and metrics
        completed_sections: Dict[str, str] = {} # Stores content of completed sections by title
        section_results: Dict[str, SectionResult] = {} # Stores full SectionResult objects
        total_metrics = WritingMetrics()
        total_article_start_time = datetime.now()

        # CORRECTED LOGIC: Process sections sequentially to ensure cohesion
        for section in sorted_sections:
            try:
                # The completed_sections dict now contains all prior sections,
                # ensuring context is passed correctly to the next section.
                result = self._write_section_with_enhanced_context(
                    article_plan, section, style_analysis, completed_sections
                )
                section_results[section.title] = result
                completed_sections[section.title] = result.content # Add to context for the *next* section
                
                # Aggregate metrics
                total_metrics.word_count += result.metrics.word_count
                total_metrics.refinement_iterations += result.metrics.refinement_iterations

            except Exception as exc:
                print(f"  ❌ Section '{section.title}' failed: {exc}")
                traceback.print_exc()
                section_results[section.title] = SectionResult(
                    heading=section.title,
                    content=f"Error generating content for '{section.title}': {exc}",
                    metrics=WritingMetrics(),
                    sources_used=[],
                    refinement_history=[f"Failed: {exc}"],
                    final_score=0.0
                )
        
        total_metrics.processing_time = (datetime.now() - total_article_start_time).total_seconds()

        # Assemble final article
        final_article = self._assemble_final_article(article_plan, section_results)
        
        # Generate article metadata
        metadata = {
            "plan": asdict(article_plan),
            "section_results": {k: asdict(v) for k, v in section_results.items()},
            "total_metrics": asdict(total_metrics),
            "generation_timestamp": datetime.now().isoformat(),
            "style_analysis": style_analysis
        }
        
        print(f"\n✅ Article completed: {total_metrics.word_count} words in {total_metrics.processing_time:.1f}s")
        return final_article, metadata

    def _sort_sections_by_dependencies(self, sections: List[SectionPlan]) -> List[SectionPlan]:
        """Sort sections to respect dependencies and priorities using a topological sort-like approach."""
        sorted_list = []
        # Create a dictionary for quick lookup of sections by title
        section_map = {sec.title: sec for sec in sections}
        # Keep track of sections that are ready to be processed (dependencies met)
        ready_queue = []
        # Keep track of remaining sections and their unresolved dependencies count
        in_degree = {sec.title: 0 for sec in sections}
        # Adjacency list to find sections that depend on a given section
        graph = {sec.title: [] for sec in sections}

        # Calculate in-degrees and build graph
        for sec in sections:
            for dep_title in sec.dependencies:
                dep_title_str = str(dep_title) # Ensure string for consistent keys
                if dep_title_str in section_map: # Only count if dependency exists
                    in_degree[sec.title] += 1
                    graph[dep_title_str].append(sec.title)
                else:
                    # FIX: Log this more clearly as a potential LLM generation issue
                    print(f"⚠️ Warning: Section '{sec.title}' depends on non-existent section '{dep_title_str}'. This might be an LLM generation error. Ignoring this dependency.")
        
        # Initialize ready_queue with sections that have no dependencies
        for sec in sections:
            if in_degree[sec.title] == 0:
                ready_queue.append(sec)
        
        # Sort ready_queue by priority (lower number = higher priority)
        ready_queue.sort(key=lambda s: s.priority)

        while ready_queue:
            current_sec = ready_queue.pop(0) # Get the highest priority ready section
            sorted_list.append(current_sec)

            # For all sections that depend on the current_sec
            for dependent_sec_title in graph[current_sec.title]:
                in_degree[dependent_sec_title] -= 1
                if in_degree[dependent_sec_title] == 0:
                    # If all dependencies are met, add to ready queue and re-sort
                    ready_queue.append(section_map[dependent_sec_title])
                    ready_queue.sort(key=lambda s: s.priority) # Re-sort after adding new ready sections
        
        # Check for circular dependencies or unreachable sections
        if len(sorted_list) != len(sections):
            unreachable_sections = [sec.title for sec in sections if sec.title not in [s.title for s in sorted_list]]
            print(f"❌ Circular dependencies or unresolvable dependencies detected. Some sections might be skipped or processed out of logical order: {unreachable_sections}")
            # As a fallback, append any remaining sections that couldn't be sorted
            # This ensures all sections are attempted, even if dependencies are problematic.
            remaining_sections_unsorted = [sec for sec in sections if sec not in sorted_list]
            # Sort remaining by priority to give some order
            remaining_sections_unsorted.sort(key=lambda s: s.priority)
            sorted_list.extend(remaining_sections_unsorted)

        return sorted_list

    def _group_sections_by_dependency_level(self, sections: List[SectionPlan]) -> Dict[int, List[SectionPlan]]:
        """Group sections by their dependency depth level using a recursive approach with memoization."""
        levels = {}
        section_levels = {} # Memoization for calculated levels
        section_map = {s.title: s for s in sections} # Map for quick lookup

        def get_level(section_title: str, visited: set):
            # FIX: Initialize 'level' to a default value to prevent UnboundLocalError
            level = 0 

            # Check for circular dependency in current path
            if section_title in visited:
                return float('inf') # Indicate circular dependency

            # Check memoization cache
            if section_title in section_levels:
                return section_levels[section_title]

            visited.add(section_title) # Mark as visited in current path

            section = section_map.get(section_title)
            if not section:
                # This case should ideally not happen if sections list is valid
                print(f"⚠️ Warning: Attempted to get level for non-existent section '{section_title}'.")
                level = 0
            elif not section.dependencies:
                level = 0 # Base case: no dependencies
            else:
                max_dep_level = -1
                for dep_title in section.dependencies:
                    dep_level = get_level(str(dep_title), visited.copy()) # Pass a copy of visited for each branch
                    if dep_level == float('inf'): # Propagate circular dependency
                        level = float('inf')
                        break
                    if dep_level > max_dep_level:
                        max_dep_level = dep_level
                if level != float('inf'): # If no circular dependency detected in dependencies
                    level = max_dep_level + 1

            visited.remove(section_title) # Remove from current path after processing
            section_levels[section_title] = level # Memoize the calculated level
            return level

        for s in sections:
            level = get_level(s.title, set())
            if level == float('inf'):
                print(f"⚠️ Circular dependency detected for section '{s.title}'. Assigning to level 0 to ensure processing.")
                level = 0 # Assign to level 0 to ensure it gets processed
            
            if level not in levels:
                levels[level] = []
            levels[level].append(s)
        
        # Sort sections within each level by priority
        for level_num in levels:
            levels[level_num].sort(key=lambda s: s.priority)

        return levels

    def _assemble_final_article(self, article_plan: ArticlePlan, 
                                section_results: Dict[str, SectionResult]) -> str:
        """Assemble the final article with proper formatting and structure."""
        print("     📄 Assembling final article...")
        
        article_lines = [f"# {article_plan.topic}", ""]
        
        # Ensure sections are assembled in the planned order,
        # but use content from section_results
        for section_plan in article_plan.sections:
            if section_plan.title in section_results:
                result = section_results[section_plan.title]
                article_lines.extend([
                    f"## {result.heading}",
                    "",
                    result.content,
                    ""
                ])
            else:
                print(f"   ⚠️ Warning: Content for section '{section_plan.title}' was not found. Skipping.")
                article_lines.extend([
                    f"## {section_plan.title}",
                    "",
                    f"*(Content for this section could not be generated or found.)*",
                    ""
                ])
        
        return "\n".join(article_lines).strip()

    def print_article_metrics(self, metadata: Dict[str, any]) -> None:
        """Print detailed metrics about the article generation process."""
        print(f"\n{'='*60}")
        print(f"📊 ARTICLE GENERATION METRICS")
        print(f"{'='*60}")
        
        total_metrics = metadata['total_metrics']
        print(f"📝 Total word count: {total_metrics['word_count']}")
        print(f"⏱️   Total processing time: {total_metrics['processing_time']:.1f}s")
        print(f"🔄 Total refinement iterations (across all sections): {total_metrics['refinement_iterations']}")
        
        print(f"\n📋 Section Performance:")
        for section_title, result_data in metadata['section_results'].items():
            metrics = result_data['metrics']
            print(f"   • {section_title}:")
            print(f"     - Words: {metrics['word_count']}")
            print(f"     - Time: {metrics['processing_time']:.1f}s")
            print(f"     - Score: {result_data['final_score']:.1f}/5.0")
            print(f"     - Refinements: {metrics['refinement_iterations']}")
            if result_data.get('refinement_history'):
                print(f"     - History: {result_data['refinement_history']}")
            if result_data.get('sources_used'):
                print(f"     - Sources (top 3): {'; '.join(result_data['sources_used'])}")
        
        print(f"\n🎯 Article Plan Summary:")
        plan = metadata['plan']
        print(f"   • Topic: {plan['topic']}")
        print(f"   • Target audience: {plan['target_audience']}")
        print(f"   • Writing style: {plan['writing_style']}")
        print(f"   • Estimated time: {plan['estimated_time']}")
        print(f"   • Sections planned: {len(plan['sections'])}")
        print(f"   • Key themes: {', '.join(plan['key_themes'])}")
        
        print(f"{'='*60}\n")

# Example usage and testing
if __name__ == '__main__':
    # Create a dummy style file for testing if it doesn't exist
    if not os.path.exists("style_sample.txt"):
        with open("style_sample.txt", "w", encoding='utf-8') as f:
            f.write("This is a sample of a journalistic writing style. It is clear, concise, and direct. It focuses on presenting facts and information to the reader in an accessible way. It often uses short paragraphs and active voice. The aim is to inform and engage the general public.")
        print("Created dummy 'style_sample.txt' for testing.")

    # Create dummy vector_store directory and files for RAGContext if they don't exist
    vector_store_dir = 'vector_store'
    if not os.path.exists(vector_store_dir):
        os.makedirs(vector_store_dir)
        print(f"Created dummy '{vector_store_dir}' directory.")
    
    # Create dummy FAISS index and chunks if they don't exist
    dummy_index_path = os.path.join(vector_store_dir, 'faiss_index.bin')
    dummy_chunks_path = os.path.join(vector_store_dir, 'chunks.pkl')
    
    if not os.path.exists(dummy_index_path) or not os.path.exists(dummy_chunks_path):
        print("Creating dummy FAISS index and chunks for testing...")
        try:
            # Create dummy embeddings and chunks
            dummy_embeddings = np.random.rand(100, 384).astype('float32') # 100 chunks, 384 dim (common for mxbai-embed)
            dummy_chunks = [f"This is a dummy chunk of information about AI in healthcare, ID {i}. It talks about diagnostics and patient care." for i in range(100)]
            
            # Create a dummy FAISS index
            dummy_index = faiss.IndexFlatL2(dummy_embeddings.shape[1])
            dummy_index.add(dummy_embeddings)
            faiss.write_index(dummy_index, dummy_index_path)
            
            with open(dummy_chunks_path, 'wb') as f:
                pickle.dump(dummy_chunks, f)
            print("Dummy FAISS index and chunks created.")
        except Exception as e:
            print(f"Error creating dummy FAISS index/chunks: {e}")
            print("Please ensure faiss-cpu is installed (`pip install faiss-cpu`) and try again.")
            sys.exit(1)


    try:
        # Initialize components
        rag_context = RAGContext() # This will now use the dummy files if real ones aren't present
        writer = WriterAgent(rag_context)
        
        # Test topics
        test_topics = [
            "The Future of Artificial Intelligence in Healthcare",
        ]
        
        for topic in test_topics:
            print(f"\n🎯 Testing enhanced writer with topic: {topic}")
            
            # Generate enhanced plan
            article_plan = writer.generate_enhanced_plan(
                topic=topic,
                target_word_count=1500,
                writing_style=WritingStyle.JOURNALISTIC,
                target_audience="tech-savvy professionals"
            )
            
            if not article_plan:
                print(f"❌ Failed to generate plan for: {topic}")
                continue
            
            print(f"✅ Plan generated with {len(article_plan.sections)} sections")
            
            # Print plan summary
            print(f"\n📋 Article Plan:")
            print(f"   Topic: {article_plan.topic}")
            print(f"   Target words: {article_plan.target_word_count}")
            print(f"   Style: {article_plan.writing_style.value}")
            print(f"   Estimated time: {article_plan.estimated_time}")
            print(f"   Sections:")
            for i, section in enumerate(article_plan.sections, 1):
                dep_strs = [str(dep) for dep in section.dependencies] if section.dependencies else []
                deps = f" (depends on: {', '.join(dep_strs)})" if dep_strs else ""
                print(f"     {i}. {section.title} [{section.section_type.value}]{deps}")
                print(f"         - Target words: {section.target_word_count}")
                print(f"         - Key points: {', '.join(section.key_points)}")
            
            # Write the article from the plan
            article, metadata = writer.write_article_from_enhanced_plan(article_plan, "style_sample.txt")
            
            # Print metrics
            writer.print_article_metrics(metadata)

            # Save the article to a file
            # Sanitize topic for filename
            sanitized_topic = re.sub(r'[^\w\s-]', '', topic).replace(' ', '_')
            output_filename = f"{sanitized_topic.lower()}_article.md"
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(article)
            print(f"✅ Article saved to {output_filename}")

            print(f"✅ Enhanced planning and writing test completed for: {topic}")
            
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}. Please ensure you have run the RAG builder script to create the vector store or that dummy files are correctly generated.")
        traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing:")
        traceback.print_exc()
