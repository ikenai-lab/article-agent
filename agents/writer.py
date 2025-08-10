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
from typing import Dict, List, Optional, Tuple, Union, Any
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
    tone: str
    dependencies: List[str] = field(default_factory=list)
    research_queries: List[str] = field(default_factory=list)
    priority: int = 1
    transition_hints: List[str] = field(default_factory=list)  # NEW: Hints for transitions
    connection_points: List[str] = field(default_factory=list) # NEW: Points to connect with other sections

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
    narrative_thread: str = ""  # NEW: Overall narrative thread
    key_arguments: List[str] = field(default_factory=list)  # NEW: Main arguments to maintain

@dataclass
class SectionResult:
    """Result of section generation with metadata."""
    heading: str
    content: str
    metrics: WritingMetrics
    sources_used: List[str]
    refinement_history: List[str]
    final_score: float
    key_concepts: List[str] = field(default_factory=list) # NEW: Key concepts for coherence tracking

@dataclass
class ArticleContext:
    """Maintains coherence context throughout article generation."""
    established_concepts: Dict[str, str] = field(default_factory=dict)  # concept -> definition
    narrative_progression: List[str] = field(default_factory=list)
    section_summaries: Dict[str, str] = field(default_factory=dict)  # section -> brief summary
    key_evidence: List[str] = field(default_factory=list) # Important evidence to reference

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

        try:
            self.embedding_model = SentenceTransformer(embed_model_name)
            self.reranker = CrossEncoder(reranker_model_name)
            print("Embedding and Reranker models loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding or reranker models: {e}")

        self._query_cache = {}
        self._lock = threading.Lock()

        print("✅ Enhanced RAG Context ready.")

    def query(self, query_text: str, initial_k: int = 15, final_top_n: int = 5,
              use_cache: bool = True) -> Tuple[List[str], List[float]]:
        query_hash = hashlib.md5(f"{query_text}_{initial_k}_{final_top_n}".encode()).hexdigest()

        if use_cache:
            with self._lock:
                if query_hash in self._query_cache:
                    print(f"   -> Cache hit for query: '{query_text[:50]}...'")
                    return self._query_cache[query_hash]

        query_embedding = self.embedding_model.encode([query_text], convert_to_tensor=True).cpu().numpy().reshape(1, -1)

        if self.index.ntotal == 0:
            print("   -> FAISS index is empty. No retrieval possible.")
            return [], []

        similarities, indices = self.index.search(query_embedding, initial_k)

        combined_faiss_results = []
        for i in range(min(initial_k, len(indices[0]))):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.chunks):
                combined_faiss_results.append((similarities[0][i], idx))

        combined_faiss_results.sort(key=lambda x: x[0], reverse=True)

        retrieved_chunks = [self.chunks[idx] for sim, idx in combined_faiss_results[:initial_k]]
        base_similarities = [sim for sim, idx in combined_faiss_results[:initial_k]]

        if not retrieved_chunks:
            print("   -> No valid chunks retrieved after initial FAISS search or filtering.")
            return [], []

        pairs = [[query_text, chunk] for chunk in retrieved_chunks]

        try:
            rerank_scores = self.reranker.predict(pairs)
            if rerank_scores.ndim > 1:
                rerank_scores = rerank_scores.flatten()

            if len(rerank_scores) != len(base_similarities):
                print(f"   CRITICAL RAG ERROR: Reranker scores length mismatch with base similarities.")
                min_len = min(len(base_similarities), len(rerank_scores))
                base_similarities = base_similarities[:min_len]
                rerank_scores = rerank_scores[:min_len]

            combined_scores = 0.3 * np.array(base_similarities) + 0.7 * np.array(rerank_scores)
        except Exception as e:
            print(f"   CRITICAL RAG ERROR: Reranker prediction failed. Error: {e}")
            traceback.print_exc()
            combined_scores = np.array(base_similarities)

        if combined_scores.size == 0:
            return [], []

        scored_chunks = list(zip(combined_scores.tolist(), retrieved_chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        top_chunks = [chunk for score, chunk in scored_chunks[:final_top_n]]
        top_scores = [float(score) for score, chunk in scored_chunks[:final_top_n]]

        result = (top_chunks, top_scores)

        if use_cache:
            with self._lock:
                self._query_cache[query_hash] = result

        return result

    def get_contextual_chunks(self, queries: List[str], max_chunks: int = 8) -> Tuple[List[str], Dict[str, float]]:
        all_chunks = []
        chunk_scores = {}

        chunks_per_query = max(1, max_chunks // len(queries)) if queries else max_chunks

        for query in queries:
            chunks, scores = self.query(query, final_top_n=chunks_per_query + 2)
            for chunk, score in zip(chunks, scores):
                if chunk not in chunk_scores or score > chunk_scores[chunk]:
                    chunk_scores[chunk] = score
                    if chunk not in all_chunks:
                        all_chunks.append(chunk)

        sorted_chunks = sorted(all_chunks, key=lambda x: chunk_scores[x], reverse=True)
        return sorted_chunks[:max_chunks], {chunk: chunk_scores[chunk] for chunk in sorted_chunks[:max_chunks]}


class WriterAgent:
    """Enhanced writer agent with sophisticated planning and quality control."""

    def __init__(self, rag_context: RAGContext, model_name="granite3.3-ctx", critique_model_name="granite3.3-ctx"):
        print("✍️ Initializing Enhanced Writer Agent...")
        self.rag_context = rag_context
        self.critique_agent = CritiqueAgent(model_name=critique_model_name)
        self.model_name = model_name

        self.max_refinements = 3
        self.min_quality_threshold = 3.5

        self.article_context = ArticleContext()
        self.writing_history = []

        try:
            ollama.show(self.model_name)
            print(f"✅ Ollama model '{self.model_name}' found for WriterAgent.")
        except Exception:
            print(f"❌ Ollama model '{self.model_name}' not found for WriterAgent.")
            raise

    def _generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
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

    def generate_enhanced_plan(self, topic: str, target_word_count: int = 2000,
                               writing_style: WritingStyle = WritingStyle.JOURNALISTIC,
                               target_audience: str = "general public",
                               suggested_outline: Optional[List[str]] = None) -> Optional[ArticlePlan]:
        print(f"📋 Generating enhanced article plan for: '{topic}'...")

        valid_section_types = "|".join([st.value for st in SectionType if st != SectionType.DEFAULT])

        outline_guidance = ""
        if suggested_outline:
            outline_list_str = "\n".join([f"- {item}" for item in suggested_outline])
            outline_guidance = f"""
SUGGESTED HIGH-LEVEL OUTLINE (from PlannerAgent):
Use these as a strong guide for your "sections" titles and overall structure.
{outline_list_str}
"""
        planning_prompt = f"""
You are an expert content strategist creating a comprehensive article plan that ensures coherence and flow.

ARTICLE REQUIREMENTS:
- Topic: "{topic}"
- Target word count: {target_word_count} words
- Writing style: {writing_style.value}
- Target audience: {target_audience}

{outline_guidance}

Create a detailed plan following this EXACT JSON structure. Ensure the final output is a single, valid JSON object.
{{
  "topic": "{topic}",
  "target_word_count": {target_word_count},
  "writing_style": "{writing_style.value}",
  "target_audience": "{target_audience}",
  "key_themes": ["theme1", "theme2", "theme3"],
  "narrative_thread": "The overarching story or argument that connects all sections",
  "key_arguments": ["main argument 1", "main argument 2"],
  "sections": [
    {{
      "title": "Section Title",
      "section_type": "{valid_section_types}",
      "key_points": ["point1", "point2", "point3"],
      "target_word_count": 400,
      "dependencies": [],
      "research_queries": ["query1", "query2"],
      "tone": "engaging|analytical|informative|persuasive|neutral",
      "priority": 1,
      "transition_hints": ["How this section connects to the next"],
      "connection_points": ["Key concepts that link to other sections"]
    }}
  ],
  "research_requirements": ["specific research needs"],
  "success_criteria": ["criteria for successful article"],
  "estimated_time": "time estimate (e.g., '1-2 hours')"
}}

PLANNING GUIDELINES:
1. Create 5-7 sections with a logical, coherent flow.
2. Ensure each section advances the 'narrative_thread' and supports the 'key_arguments'.
3. Include 'transition_hints' that explain how each section connects to the next.
4. Add 'connection_points' that identify recurring themes and concepts.
5. If a SUGGESTED OUTLINE is provided, use its titles as the primary basis for your "sections" titles.

Provide ONLY the JSON object.
"""

        response = self._generate_response(planning_prompt, temperature=0.5)

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON object found in response", response, 0)

            json_string = json_match.group(0)
            json_string = re.sub(r",\s*([}\]])", r"\1", json_string)
            plan_dict = json.loads(json_string)

            sections = []
            parsed_section_titles = {s_data.get('title') for s_data in plan_dict.get('sections', []) if s_data.get('title')}

            for section_data in plan_dict.get('sections', []):
                section_type_str_raw = section_data.get('section_type', 'analysis').upper()
                found_section_type = SectionType.DEFAULT
                for st_enum in SectionType:
                    if st_enum != SectionType.DEFAULT and st_enum.value.upper() in section_type_str_raw:
                        found_section_type = st_enum
                        break

                section = SectionPlan(
                    title=section_data.get('title', 'Untitled Section'),
                    section_type=found_section_type,
                    key_points=section_data.get('key_points', []),
                    target_word_count=section_data.get('target_word_count', 300),
                    tone=section_data.get('tone', 'informative'),
                    dependencies=[str(dep).strip() for dep in section_data.get('dependencies', []) if str(dep).strip() in parsed_section_titles],
                    research_queries=section_data.get('research_queries', []),
                    priority=section_data.get('priority', 1),
                    transition_hints=section_data.get('transition_hints', []),
                    connection_points=section_data.get('connection_points', [])
                )
                sections.append(section)

            validated_writing_style = WritingStyle[plan_dict.get('writing_style', writing_style.value).upper()]

            article_plan = ArticlePlan(
                topic=plan_dict.get('topic', topic),
                target_word_count=plan_dict.get('target_word_count', target_word_count),
                writing_style=validated_writing_style,
                target_audience=plan_dict.get('target_audience', target_audience),
                key_themes=plan_dict.get('key_themes', []),
                sections=sections,
                research_requirements=plan_dict.get('research_requirements', []),
                success_criteria=plan_dict.get('success_criteria', []),
                estimated_time=plan_dict.get('estimated_time', 'Not specified'),
                narrative_thread=plan_dict.get('narrative_thread', ''),
                key_arguments=plan_dict.get('key_arguments', [])
            )

            print("✅ Enhanced article plan generated successfully.")
            return article_plan

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"❌ Failed to parse article plan: {e}")
            traceback.print_exc()
            return None

    def _build_coherence_context(self, article_plan: ArticlePlan) -> str:
        """Builds a string of the current article context to ensure coherence."""
        context_parts = [
            f"NARRATIVE THREAD: {article_plan.narrative_thread}",
            f"KEY ARGUMENTS TO MAINTAIN: {', '.join(article_plan.key_arguments)}"
        ]
        if self.article_context.section_summaries:
            summaries = "\n".join(f"- {title}: {summary}" for title, summary in self.article_context.section_summaries.items())
            context_parts.append(f"SUMMARY OF PREVIOUS SECTIONS:\n{summaries}")
        if self.article_context.established_concepts:
            concepts = "\n".join(f"- {concept}: {definition}" for concept, definition in self.article_context.established_concepts.items())
            context_parts.append(f"ESTABLISHED CONCEPTS (Do not redefine, but refer to them):\n{concepts}")

        return "\n\n".join(context_parts)

    def _update_article_context(self, section_title: str, content: str):
        """Extracts key info from content and updates the shared ArticleContext."""
        summary_prompt = f"Summarize the following text in one sentence for context in a larger article:\n\n{content[:1000]}"
        summary = self._generate_response(summary_prompt, temperature=0.2, max_tokens=150)
        self.article_context.section_summaries[section_title] = summary.strip()

        concepts_prompt = f"List the 3-5 most important new concepts or terms introduced in this text, separated by commas:\n\n{content[:1500]}"
        concepts_str = self._generate_response(concepts_prompt, temperature=0.2, max_tokens=100)
        new_concepts = [c.strip() for c in concepts_str.split(',') if c.strip()]
        for concept in new_concepts:
            if concept not in self.article_context.established_concepts:
                self.article_context.established_concepts[concept] = f"Introduced in '{section_title}'"

    def _compress_and_synthesize_context(self, chunks: List[str], confidence_scores: Dict[str, float],
                                         section: SectionPlan, max_synthesis_words: int = 400) -> str:
        print("     -> Synthesizing context for section requirements...")
        if not chunks:
            return "No research context was found for this section."

        # Weight chunks based on relevance to key points and connection points
        weighted_chunks = []
        for chunk in chunks:
            score = confidence_scores.get(chunk, 0.0)
            relevance_bonus = sum(0.1 for point in section.key_points if any(word.lower() in chunk.lower() for word in point.split()))
            relevance_bonus += sum(0.15 for point in section.connection_points if any(word.lower() in chunk.lower() for word in point.split()))
            weighted_chunks.append((chunk, score + relevance_bonus))

        weighted_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks_for_synthesis = [chunk for chunk, _ in weighted_chunks[:8]]

        synthesis_prompt = f"""
Synthesize the following research chunks into a coherent summary focused on these key points: {', '.join(section.key_points)}

RESEARCH CHUNKS:
{"---CHUNK---".join(top_chunks_for_synthesis)}

SYNTHESIS REQUIREMENTS:
1. Focus on information relevant to: {', '.join(section.key_points)}
2. Organize facts logically.
3. Preserve specific data, numbers, and quotes.
4. Target a concise summary around {max_synthesis_words} words.
5. CRITICAL: Output only the synthesis.

SYNTHESIS:
"""
        return self._generate_response(synthesis_prompt, temperature=0.4, max_tokens=int(max_synthesis_words * 1.2))

    def _write_section_with_enhanced_context(self, article_plan: ArticlePlan, section: SectionPlan,
                                             style_profile: Dict[str, Any],
                                             completed_sections: Dict[str, str]) -> SectionResult:
        start_time = datetime.now()
        print(f"\n   🎯 Writing section: '{section.title}' ({section.section_type.value})")

        all_research_queries = section.research_queries + [f"{article_plan.topic} {section.title}"] + section.connection_points
        research_chunks, confidence_scores_dict = self.rag_context.get_contextual_chunks(
            all_research_queries, max_chunks=10
        )

        synthesized_context = self._compress_and_synthesize_context(
            research_chunks, confidence_scores_dict, section
        )

        # Build dependency and coherence contexts
        dependency_context = "\n\n".join([f"--- Content from '{title}' ---{content[:500]}..." for title, content in completed_sections.items() if title in section.dependencies])
        coherence_context = self._build_coherence_context(article_plan)

        style_profile_str = json.dumps(style_profile, indent=2)

        section_prompt = f"""
You are an expert writer crafting a specific section of an article. Follow these requirements precisely:

ARTICLE CONTEXT:
- Overall topic: "{article_plan.topic}"
- Target audience: {article_plan.target_audience}
- Section title: "{section.title}"
- Target word count for this section: {section.target_word_count} words

KEY POINTS TO ADDRESS IN THIS SECTION:
{chr(10).join(f"• {point}" for point in section.key_points)}

WRITING STYLE REQUIREMENTS (You must follow these rules):
---
{style_profile_str}
---

COHERENCE CONTEXT (CRITICAL: maintain consistency with this):
---
{coherence_context}
---

RESEARCH CONTEXT (Use ONLY this information for facts):
---
{synthesized_context}
---

DEPENDENCY CONTEXT (Content from previous sections, do NOT repeat):
---
{dependency_context if dependency_context else "No dependency context available."}
---

WRITING INSTRUCTIONS:
1.  Write ONLY the body content for this section. Do NOT repeat the section title.
2.  **COHERENCE IS PRIORITY**: Your goal is to build upon the COHERENCE CONTEXT. Advance the narrative. Do NOT repeat facts or themes already covered.
3.  Address all key points using the RESEARCH CONTEXT.
4.  Weave in themes from the 'connection_points': {', '.join(section.connection_points)}.
5.  End the section with a sentence that hints at the next topic, using these hints: {', '.join(section.transition_hints)}.
6.  **CRITICAL**: **DO NOT** quote from the original style sample. Follow the style rules to create original content.

CRITICAL: Your response must contain ONLY the article section's content.
"""

        draft_content = self._generate_response(section_prompt, temperature=0.7, max_tokens=int(section.target_word_count * 2.0))

        metrics = WritingMetrics()
        refinement_history = []
        sources_used = [chunk[:100] + "..." for chunk in research_chunks[:3]]
        final_score = 0.0
        for iteration in range(self.max_refinements):
            print(f"     -> Running critique for iteration {iteration + 1}...")
            critique_result = self.critique_agent.critique_section(
                article_plan.topic, section.title, draft_content,
                synthesized_context, json.dumps(style_profile),
                coherence_context # Add this new argument
            )
            current_score = critique_result.get('score', 0.0)
            critiques = critique_result.get('critique', [])

            if current_score >= self.min_quality_threshold:
                final_score = current_score
                break
            elif critiques and iteration < self.max_refinements - 1:
                refinement_history.append(f"Iteration {iteration + 1}: {critiques}")
                draft_content = self._rewrite_section_with_enhanced_feedback(article_plan, section, draft_content, critiques, style_profile, synthesized_context)
            else:
                final_score = current_score
                break

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
                                                style_profile: Dict[str, Any], context: str) -> str:
        critique_points = "\n".join(f"• {critique}" for critique in critiques)
        style_profile_str = json.dumps(style_profile, indent=2)
        rewrite_prompt = f"""
Revise a section of an article based on editorial feedback.

ORIGINAL ARTICLE TOPIC: "{article_plan.topic}"
SECTION TITLE: "{section.title}"
CURRENT DRAFT:
---
{draft_content}
---

EDITORIAL FEEDBACK TO ADDRESS:
{critique_points}

REQUIRED WRITING STYLE:
---
{style_profile_str}
---

SUPPORTING CONTEXT (for factual accuracy):
---
{context[:1000]}...
---

REVISION INSTRUCTIONS:
1. Address each point in the editorial feedback specifically.
2. Maintain the target word count ({section.target_word_count} words).
3. Preserve the required writing style.
4. **DO NOT** quote from the original style sample.

Provide ONLY the complete, revised section content.
"""
        return self._generate_response(rewrite_prompt, temperature=0.6, max_tokens=int(section.target_word_count * 2.0))

    def write_article_from_enhanced_plan(self, article_plan: ArticlePlan, style_profile: Dict[str, Any]) -> Tuple[str, Dict[str, any], Dict[str, int]]:
        print(f"\n✍️ Writing article: '{article_plan.topic}'")
        sorted_sections = self._sort_sections_by_dependencies(article_plan.sections)
        print(f"  Order of sections: {[s.title for s in sorted_sections]}")

        completed_sections: Dict[str, str] = {}
        section_results: Dict[str, SectionResult] = {}
        total_metrics = WritingMetrics()
        total_article_start_time = datetime.now()
        # Initialize a dictionary to hold token counts
        writer_tokens = {'prompt_tokens': 0, 'eval_tokens': 0, 'total_tokens': 0}

        for section in sorted_sections:
            try:
                result = self._write_section_with_enhanced_context(
                    article_plan, section, style_profile, completed_sections
                )
                section_results[section.title] = result
                completed_sections[section.title] = result.content

                self._update_article_context(section.title, result.content)

                total_metrics.word_count += result.metrics.word_count
                total_metrics.refinement_iterations += result.metrics.refinement_iterations

            except Exception as exc:
                print(f"   ❌ Section '{section.title}' failed: {exc}")
                traceback.print_exc()
                section_results[section.title] = SectionResult(heading=section.title, content=f"Error: {exc}", metrics=WritingMetrics(), sources_used=[], refinement_history=[], final_score=0.0)

        total_metrics.processing_time = (datetime.now() - total_article_start_time).total_seconds()
        final_article = self._assemble_final_article(article_plan, section_results)
        metadata = {
            "plan": asdict(article_plan),
            "section_results": {k: asdict(v) for k, v in section_results.items()},
            "total_metrics": asdict(total_metrics),
            "generation_timestamp": datetime.now().isoformat(),
            "style_analysis": style_profile
        }

        print(f"\n✅ Article completed: {total_metrics.word_count} words in {total_metrics.processing_time:.1f}s")
        return final_article, metadata, writer_tokens

    def _sort_sections_by_dependencies(self, sections: List[SectionPlan]) -> List[SectionPlan]:
        sorted_list = []
        section_map = {sec.title: sec for sec in sections}
        in_degree = {sec.title: len(sec.dependencies) for sec in sections}
        queue = [sec for sec in sections if in_degree[sec.title] == 0]

        while queue:
            current = queue.pop(0)
            sorted_list.append(current)
            for other_sec in sections:
                if current.title in other_sec.dependencies:
                    in_degree[other_sec.title] -= 1
                    if in_degree[other_sec.title] == 0:
                        queue.append(other_sec)

        if len(sorted_list) != len(sections):
            unreachable = [sec.title for sec in sections if sec.title not in [s.title for s in sorted_list]]
            print(f"⚠️ Circular dependency detected or missing dependency. Unreachable sections: {unreachable}")
            sorted_list.extend([sec for sec in sections if sec.title not in [s.title for s in sorted_list]])

        return sorted_list


    def _assemble_final_article(self, article_plan: ArticlePlan,
                                section_results: Dict[str, SectionResult]) -> str:
        print("     📄 Assembling final article...")
        article_lines = [f"# {article_plan.topic}", ""]
        for section_plan in article_plan.sections:
            if section_plan.title in section_results:
                result = section_results[section_plan.title]
                article_lines.extend([f"## {result.heading}", "", result.content, ""])
            else:
                article_lines.extend([f"## {section_plan.title}", "", "*(Content could not be generated.)*", ""])
        return "\n".join(article_lines).strip()

    def print_article_metrics(self, metadata: Dict[str, any]) -> None:
        print(f"\n{'='*60}")
        print(f"📊 ARTICLE GENERATION METRICS")
        print(f"{'='*60}")
        total_metrics = metadata['total_metrics']
        print(f"📝 Total word count: {total_metrics['word_count']}")
        print(f"⏱️   Total processing time: {total_metrics['processing_time']:.1f}s")
        print(f"🔄 Total refinement iterations: {total_metrics['refinement_iterations']}")
        print(f"\n📋 Section Performance:")
        for section_title, result_data in metadata['section_results'].items():
            metrics = result_data['metrics']
            print(f"   • {section_title}: Words: {metrics['word_count']}, Time: {metrics['processing_time']:.1f}s, Score: {result_data['final_score']:.1f}/5.0")
        print(f"\n🎯 Article Plan Summary:")
        plan = metadata['plan']
        print(f"   • Topic: {plan['topic']}")
        print(f"   • Target audience: {plan['target_audience']}")
        print(f"{'='*60}\n")