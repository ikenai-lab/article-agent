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
    from agents.ollama_token_counter import chat_with_token_counts
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
    DEFAULT = "default"

@dataclass
class WritingMetrics:
    """Metrics for tracking writing quality and performance."""
    word_count: int = 0
    readability_score: float = 0.0
    coherence_score: float = 0.0
    style_consistency: float = 0.0
    factual_accuracy: float = 0.0
    engagement_score: float = 0.0
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
    transition_hints: List[str] = field(default_factory=list)
    connection_points: List[str] = field(default_factory=list)

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
    narrative_thread: str = ""
    key_arguments: List[str] = field(default_factory=list)

@dataclass
class SectionResult:
    """Result of section generation with metadata."""
    heading: str
    content: str
    metrics: WritingMetrics
    sources_used: List[str]
    refinement_history: List[str]
    final_score: float
    key_concepts: List[str] = field(default_factory=list)

@dataclass
class ArticleContext:
    """Maintains coherence context throughout article generation."""
    established_concepts: Dict[str, str] = field(default_factory=dict)
    narrative_progression: List[str] = field(default_factory=list)
    section_summaries: Dict[str, str] = field(default_factory=dict)
    key_evidence: List[str] = field(default_factory=list)

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

    def _generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> Tuple[str, Dict[str, int]]:
        try:
            result_data = chat_with_token_counts(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )
            tokens = {
                'prompt_tokens': result_data.get('prompt_tokens', 0),
                'eval_tokens': result_data.get('eval_tokens', 0),
                'total_tokens': result_data.get('total_tokens', 0)
            }
            return result_data['response'], tokens
        except Exception as e:
            print(f"Error during Ollama generation: {e}")
            return "", {'prompt_tokens': 0, 'eval_tokens': 0, 'total_tokens': 0}

    def generate_enhanced_plan(self, topic: str, target_word_count: int = 2000,
                               writing_style: WritingStyle = WritingStyle.JOURNALISTIC,
                               target_audience: str = "general public",
                               suggested_outline: Optional[List[str]] = None) -> Tuple[Optional[ArticlePlan], Dict[str, int]]:
        """
        Constructs a detailed ArticlePlan directly from the PlannerAgent's narrative outline.
        This method no longer calls an LLM, removing a source of errors and redundancy.
        """
        print("📋 Translating narrative outline into detailed article plan...")
        
        if not suggested_outline:
            print("❌ Cannot generate plan: The suggested_outline from the PlannerAgent is empty.")
            return None, {}

        sections = []
        num_sections = len(suggested_outline)
        words_per_section = target_word_count // num_sections if num_sections > 0 else target_word_count

        for i, outline_item in enumerate(suggested_outline):
            parts = outline_item.split(':', 1)
            title = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else f"Discuss {title}"
            
            # Simple logic to determine section type based on order
            section_type = SectionType.INTRODUCTION
            if i > 0 and i < num_sections - 1:
                section_type = SectionType.ANALYSIS
            elif i == num_sections - 1:
                section_type = SectionType.CONCLUSION

            section = SectionPlan(
                title=title,
                section_type=section_type,
                key_points=[description],  # The description from the planner becomes the key point
                target_word_count=words_per_section,
                tone="informative", # Default tone
                dependencies= [s.title for s in sections], # Depends on all previous sections
                research_queries=[f"{topic} {title}", description],
                priority=i + 1,
                transition_hints=[f"Lead into the next topic: {suggested_outline[i+1].split(':')[0].strip()}" if i < num_sections - 1 else "Conclude the article."],
                connection_points=[] # Can be enhanced later if needed
            )
            sections.append(section)

        article_plan = ArticlePlan(
            topic=topic,
            target_word_count=target_word_count,
            writing_style=writing_style,
            target_audience=target_audience,
            key_themes=[],
            sections=sections,
            research_requirements=[],
            success_criteria=[],
            estimated_time="1-2 hours",
            narrative_thread=f"A narrative exploring the topic of {topic}.",
            key_arguments=[]
        )

        print("✅ Detailed article plan created successfully from narrative outline.")
        # No LLM call, so no token usage to return
        return article_plan, {}


    def _build_coherence_context(self, article_plan: ArticlePlan) -> str:
        """Builds a string of the current article context to ensure coherence."""
        context_parts = [
            f"NARRATIVE THREAD: {article_plan.narrative_thread}",
            f"KEY ARGUMENTS TO MAINTAIN: {', '.join(article_plan.key_arguments)}"
        ]
        if self.article_context.section_summaries:
            summaries = "\n".join(f"- {title}: {summary}" for title, summary in self.article_context.section_summaries.items())
            context_parts.append(f"SUMMARY OF PREVIOUS SECTIONS:\n{summaries}")
        if self.article_context.key_evidence:
            evidence = ", ".join(self.article_context.key_evidence)
            context_parts.append(f"FACTS & ENTITIES ALREADY MENTIONED (Do NOT re-introduce these. You can refer to them, but assume the reader already knows them): {evidence}")

        return "\n\n".join(context_parts)

    def _update_article_context(self, section_title: str, content: str):
        """Extracts key info from content and updates the shared ArticleContext."""
        summary_prompt = f"Summarize the following text in one sentence for context in a larger article:\n\n{content[:1000]}"
        summary, _ = self._generate_response(summary_prompt, temperature=0.2, max_tokens=150)
        self.article_context.section_summaries[section_title] = summary.strip()

        entities_prompt = f"""
From the following text, extract a list of key entities (companies, people, specific data points like layoff numbers or percentages) that are central to this section.
Return a comma-separated list. For example: Google, Chegg, 12,000 employees, 17%.

TEXT:
---
{content[:1500]}
---

KEY ENTITIES:
"""
        entities_str, _ = self._generate_response(entities_prompt, temperature=0.1, max_tokens=150)
        new_entities = [e.strip().lower() for e in entities_str.split(',') if e.strip()]
        self.article_context.key_evidence.extend(new_entities)
        self.article_context.key_evidence = sorted(list(set(self.article_context.key_evidence)))

    def _compress_and_synthesize_context(self, chunks: List[str], confidence_scores: Dict[str, float],
                                         section: SectionPlan, max_synthesis_words: int = 400) -> str:
        print("     -> Synthesizing context for section requirements...")
        if not chunks:
            return "No research context was found for this section."

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
        synthesis, _ = self._generate_response(synthesis_prompt, temperature=0.4, max_tokens=int(max_synthesis_words * 1.2))
        return synthesis

    def _write_section_with_enhanced_context(self, article_plan: ArticlePlan, section: SectionPlan,
                                             style_profile: Dict[str, Any],
                                             completed_sections: Dict[str, str]) -> Tuple[SectionResult, Dict[str, int]]:
        start_time = datetime.now()
        print(f"\n   🎯 Writing section: '{section.title}' ({section.section_type.value})")

        section_tokens = {'prompt_tokens': 0, 'eval_tokens': 0, 'total_tokens': 0}

        all_research_queries = section.research_queries + [f"{article_plan.topic} {section.title}"] + section.connection_points
        research_chunks, confidence_scores_dict = self.rag_context.get_contextual_chunks(
            all_research_queries, max_chunks=10
        )

        synthesized_context = self._compress_and_synthesize_context(
            research_chunks, confidence_scores_dict, section
        )

        coherence_context = self._build_coherence_context(article_plan)
        persona_profile_str = json.dumps(style_profile, indent=2)

        section_prompt = f"""
You are an expert writer tasked with embodying a specific persona to write one section of an article.

**PERSONA PROFILE (You MUST adopt this persona):**
---
{persona_profile_str}
---
Your primary goal is to write in this voice and style consistently.

**SECTION GOAL:**
- **Section Title (for your context only, do not repeat):** "{section.title}"
- **Purpose:** You need to write the body content that fulfills the narrative purpose of this section, as part of the larger article on "{article_plan.topic}".
- **Target Word Count:** Approximately {section.target_word_count} words.
- **Formatting:** If presenting a list of statistics, companies, or data points, use markdown bullet points (`* Item`) for clarity. Otherwise, write in paragraphs.

**MEMORY & COHERENCE (CRITICAL):**
You are not writing in a vacuum. Build upon what's already been written and **AVOID REPETITION.**
---
{coherence_context}
---
The facts/entities listed above are known. Your task is to introduce NEW information or analysis from the research context below.

**RESEARCH CONTEXT (Source of new facts for this section):**
---
{synthesized_context}
---

**ABSOLUTE RULES:**
1.  **DO NOT** include the section title or any markdown headings (e.g., ##) in your output.
2.  **DO NOT** write fragmented or incomplete sentences. Your output must be a complete, coherent text.
3.  **Your response must be ONLY the text content of the section body.**

**OUTPUT ONLY THE SECTION BODY CONTENT.**
"""
        draft_content, draft_tokens = self._generate_response(section_prompt, temperature=0.7, max_tokens=int(section.target_word_count * 3.5) + 300)
        section_tokens['total_tokens'] += draft_tokens['total_tokens']

        metrics = WritingMetrics()
        refinement_history = []
        sources_used = [chunk[:100] + "..." for chunk in research_chunks[:3]]
        final_score = 0.0
        for iteration in range(self.max_refinements):
            print(f"     -> Running critique for iteration {iteration + 1}...")
            critique_result = self.critique_agent.critique_section(
                article_plan.topic, section.title, draft_content,
                synthesized_context, json.dumps(style_profile),
                coherence_context, section.target_word_count
            )
            section_tokens['total_tokens'] += critique_result.get('total_tokens', 0)
            current_score = critique_result.get('score', 0.0)
            critiques = critique_result.get('critique', [])

            if current_score >= self.min_quality_threshold and not any("word count" in c.lower() for c in critiques) and not any("completeness" in c.lower() for c in critiques):
                final_score = current_score
                break
            elif critiques and iteration < self.max_refinements - 1:
                refinement_history.append(f"Iteration {iteration + 1}: {critiques}")
                draft_content, rewrite_tokens = self._rewrite_section_with_enhanced_feedback(article_plan, section, draft_content, critiques, style_profile, synthesized_context)
                section_tokens['total_tokens'] += rewrite_tokens['total_tokens']
            else:
                final_score = current_score
                break

        processing_time = (datetime.now() - start_time).total_seconds()
        metrics.word_count = len(draft_content.split())
        metrics.processing_time = processing_time
        metrics.refinement_iterations = len(refinement_history)

        result = SectionResult(
            heading=section.title,
            content=draft_content,
            metrics=metrics,
            sources_used=sources_used,
            refinement_history=refinement_history,
            final_score=final_score
        )
        return result, section_tokens

    def _rewrite_section_with_enhanced_feedback(self, article_plan: ArticlePlan, section: SectionPlan,
                                                draft_content: str, critiques: List[str],
                                                style_profile: Dict[str, Any], context: str) -> Tuple[str, Dict[str, int]]:
        critique_points = "\n".join(f"• {critique}" for critique in critiques)
        style_profile_str = json.dumps(style_profile, indent=2)
        rewrite_prompt = f"""
You are revising a single article section based on editorial feedback. You MUST maintain the persona from the style profile.

**PERSONA PROFILE (Embody this):**
---
{style_profile_str}
---

**ORIGINAL DRAFT:**
---
{draft_content}
---

**EDITORIAL FEEDBACK (You must address these points):**
{critique_points}

**SUPPORTING CONTEXT (for factual accuracy):**
---
{context[:1000]}...
---

**REVISION INSTRUCTIONS:**
1.  Address every point in the editorial feedback.
2.  Maintain the target word count ({section.target_word_count} words).
3.  Strictly adhere to the persona profile.
4.  Ensure the revised section is complete and does not end abruptly.

Provide ONLY the complete, revised section content. Do not include the title.
"""
        return self._generate_response(rewrite_prompt, temperature=0.6, max_tokens=int(section.target_word_count * 3.5) + 300)

    def write_article_from_enhanced_plan(self, article_plan: ArticlePlan, style_profile: Dict[str, Any]) -> Tuple[str, Dict[str, any], Dict[str, int]]:
        print(f"\n✍️ Writing article: '{article_plan.topic}'")
        sorted_sections = self._sort_sections_by_dependencies(article_plan.sections)
        print(f"  Order of sections: {[s.title for s in sorted_sections]}")

        completed_sections: Dict[str, str] = {}
        section_results: Dict[str, SectionResult] = {}
        total_metrics = WritingMetrics()
        total_article_start_time = datetime.now()
        writer_tokens = {'prompt_tokens': 0, 'eval_tokens': 0, 'total_tokens': 0}

        for section in sorted_sections:
            try:
                result, section_tokens = self._write_section_with_enhanced_context(
                    article_plan, section, style_profile, completed_sections
                )
                writer_tokens['total_tokens'] += section_tokens['total_tokens']

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
        # With the new planner logic, sections are already in order.
        # This function remains for robustness but is simpler.
        return sections

    def _assemble_final_article(self, article_plan: ArticlePlan,
                                section_results: Dict[str, SectionResult]) -> str:
        print("     📄 Assembling final article...")
        article_lines = [f"# {article_plan.topic}", ""]
        for section_plan in article_plan.sections:
            if section_plan.title in section_results:
                result = section_results[section_plan.title]
                clean_content = result.content.strip()

                if clean_content.lower().startswith(result.heading.lower()):
                    clean_content = clean_content[len(result.heading):].lstrip()
                    clean_content = clean_content.lstrip('#').lstrip()


                article_lines.append(f"## {result.heading}")
                article_lines.append("")
                article_lines.append(clean_content)
                article_lines.append("")
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