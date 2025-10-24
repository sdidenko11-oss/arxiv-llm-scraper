# arxiv_llm_scraper_hybrid_v2.py
# ВЕРСІЯ 2.1 - Розширений фокус на ВСЮ LLM екосистему
# FIXED: Added arxiv.Client() and error handling for UnexpectedEmptyPageError

from openai import OpenAI
import os
import csv
import json
import arxiv
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

class ArxivLLMScraperHybrid:
    def __init__(self):
        load_dotenv()
        self.arxiv_client = arxiv.Client()  # FIX: Added arxiv Client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # ЗБАЛАНСОВАНІ ключові слова - фокус на ВСЮ LLM екосистему!
        self.keyword_scores = {
            # 📚 Оглядові статті (найвища цінність!)
            "survey": 100, "review": 100, "systematic review": 95, 
            "taxonomy": 90, "comprehensive": 70, "state-of-the-art": 75,
            "literature review": 85, "overview": 65,
            
            # ✍️ PROMPT ENGINEERING (ДУЖЕ ВАЖЛИВО!)
            "prompt engineering": 70, "prompting": 65, "prompt design": 65,
            "prompt": 30,  # загальний термін
            "chain-of-thought": 65, "cot": 60, "thought": 20,
            "self-consistency": 60, "consistency": 25,
            "tree of thoughts": 65, "tot": 60, "graph of thoughts": 60,
            "react": 60, "reason and act": 60, "reasoning and acting": 55,
            "reflexion": 60, "reflection": 40, "self-reflection": 55,
            "least-to-most": 55, "decomposition": 40,
            "self-ask": 55, "maieutic prompting": 55,
            "automatic prompt engineering": 65, "ape": 55,
            "prompt optimization": 65, "prompt tuning": 60,
            "meta-prompting": 60, "prompt chaining": 55,
            "prompt decomposition": 55, "prompt selection": 50,
            "zero-shot": 50, "few-shot": 50, "one-shot": 45,
            "in-context learning": 60, "icl": 55, "contextual learning": 50,
            "role prompting": 50, "persona prompting": 45,
            "system prompt": 45, "instruction": 30,
            "demonstration": 40, "example-based": 40,
            "chain of thought prompting": 65,
            
            # 🆕 НОВІ МОДЕЛІ та АРХІТЕКТУРИ (ВАЖЛИВО!)
            "gpt-4": 55, "gpt-5": 65, "gpt4": 50, "gpt5": 60,
            "claude": 50, "claude 3": 55, "claude 4": 60,
            "claude opus": 55, "claude sonnet": 50,
            "gemini": 55, "gemini ultra": 60, "gemini pro": 55,
            "gemini advanced": 55, "gemini 2": 60,
            "llama": 50, "llama 3": 55, "llama 4": 60, "llama3": 50,
            "mistral": 50, "mixtral": 55, "mixtral 8x7b": 55,
            "palm": 45, "palm 2": 50, "bard": 40,
            "grok": 50, "grok-1": 50,
            "phi": 45, "phi-3": 50,
            "multimodal": 55, "vision-language": 50, "vlm": 50,
            "text-to-image": 40, "image generation": 40,
            "transformer": 35, "attention mechanism": 40,
            "mixture of experts": 55, "moe": 50, "sparse model": 45,
            "sparse mixture": 50, "dense model": 35,
            "architecture": 25, "model architecture": 35,
            
            # 🧠 LLM CAPABILITIES (ДУЖЕ ВАЖЛИВО!)
            "reasoning": 55, "logical reasoning": 55, "reasoning ability": 55,
            "mathematical reasoning": 60, "math reasoning": 60,
            "symbolic reasoning": 50, "abstract reasoning": 50,
            "common sense": 50, "commonsense reasoning": 55,
            "commonsense": 45, "world knowledge": 40,
            "multi-step reasoning": 55, "step-by-step": 50,
            "planning": 50, "plan generation": 50, "task planning": 50,
            "problem solving": 50, "decision making": 45,
            "code generation": 55, "program synthesis": 55,
            "code completion": 45, "code understanding": 45,
            "coding": 30, "programming": 30,
            "creative writing": 40, "story generation": 40,
            "content generation": 40, "text generation": 35,
            "summarization": 45, "summary": 25,
            "question answering": 45, "qa": 40, "q&a": 40,
            "translation": 40, "machine translation": 40,
            "multilingual": 45, "cross-lingual": 45,
            "language understanding": 40, "comprehension": 35,
            "instruction following": 55, "following instructions": 50,
            "task completion": 45, "task solving": 40,
            "dialogue": 35, "conversation": 35, "chat": 25,
            
            # 📏 CONTEXT ENGINEERING (ДУЖЕ ВАЖЛИВО!)
            "context window": 65, "context length": 60,
            "long context": 70, "extended context": 65,
            "infinite context": 70, "unlimited context": 65,
            "100k context": 65, "1m context": 70, "million token": 65,
            "context compression": 60, "context management": 55,
            "context utilization": 55, "context handling": 50,
            "memory": 45, "long-term memory": 55, "short-term memory": 45,
            "working memory": 45, "memory management": 50,
            "recurrent": 40, "recurrent model": 40,
            "state space model": 50, "ssm": 45, "mamba": 50,
            "attention": 30, "self-attention": 35,
            "efficient attention": 45, "linear attention": 45,
            
            # 🤖 AI АГЕНТИ (ВАЖЛИВО!)
            "autonomous agent": 60, "ai agent": 55, "llm agent": 60,
            "multi-agent": 60, "multiagent": 55, "agent system": 55,
            "agent framework": 55, "agentic": 50, "agent-based": 50,
            "tool-using agent": 55, "tool use": 50, "tool calling": 50,
            "function calling": 55, "api calling": 45,
            "langchain": 40, "autogpt": 45, "babyagi": 40,
            "agent communication": 50, "agent collaboration": 50,
            "reasoning agent": 55, "planning agent": 55,
            "cognitive architecture": 50, "agent architecture": 50,
            
            # 🔍 RAG & RETRIEVAL (ВАЖЛИВО!)
            "retrieval-augmented": 55, "rag": 55,
            "retrieval augmented generation": 60,
            "retrieval": 45, "knowledge retrieval": 50,
            "semantic search": 45, "dense retrieval": 45,
            "vector database": 45, "vector search": 45,
            "embedding": 40, "text embedding": 40,
            "hybrid search": 45, "reranking": 40,
            "knowledge base": 35, "external knowledge": 45,
            
            # 📊 EVALUATION & BENCHMARKS (ВАЖЛИВО!)
            "benchmark": 55, "benchmarking": 50,
            "evaluation": 50, "metric": 45, "evaluation metric": 50,
            "human evaluation": 60, "human assessment": 55,
            "llm-as-judge": 55, "llm as a judge": 55,
            "auto-evaluation": 50, "automatic evaluation": 50,
            "performance analysis": 45, "ablation study": 50,
            "empirical study": 50, "experimental": 30,
            "leaderboard": 45, "comparison": 35,
            "dataset": 30, "test set": 30,
            
            # 🎓 TRAINING & FINE-TUNING
            "instruction tuning": 55, "supervised fine-tuning": 50,
            "fine-tuning": 45, "finetuning": 45,
            "rlhf": 55, "reinforcement learning from human feedback": 60,
            "reinforcement learning": 40, "reward model": 45,
            "parameter-efficient": 50, "peft": 45,
            "lora": 50, "qlora": 50, "adapter": 45,
            "prefix tuning": 45, "prompt tuning": 60,
            "constitutional ai": 55, "alignment": 55,
            "value alignment": 50, "ai alignment": 50,
            "dpo": 50, "direct preference optimization": 55,
            "preference learning": 50, "preference optimization": 50,
            "pretraining": 35, "pre-training": 35,
            "scaling law": 50, "scaling laws": 55,
            "emergent": 50, "emergent abilities": 60,
            "emergence": 50, "emergent behavior": 55,
            
            # 🔒 SAFETY & ROBUSTNESS
            "prompt injection": 45, "injection attack": 40,
            "jailbreak": 40, "jailbreaking": 40,
            "adversarial": 45, "adversarial attack": 45,
            "adversarial prompting": 50, "red teaming": 50,
            "safety": 45, "ai safety": 50,
            "robustness": 45, "model robustness": 45,
            "bias": 40, "fairness": 40, "toxicity": 40,
            "harmful content": 40, "content moderation": 35,
            "hallucination": 50, "factuality": 50,
            "truthfulness": 45, "veracity": 40,
            
            # 🛠️ ПРАКТИЧНІ ТЕХНІКИ
            "structured output": 50, "json mode": 45,
            "json": 30, "structured generation": 50,
            "output parsing": 40, "constrained decoding": 45,
            "grammar-based generation": 45,
            
            # 🏢 INDUSTRY & APPLICATIONS
            "production": 35, "deployment": 35,
            "enterprise": 30, "real-world": 35,
            "use case": 30, "application": 25,
            "chatbot": 30, "virtual assistant": 35,
            "customer service": 30, "healthcare": 30,
            "legal": 30, "finance": 30, "education": 30,
            
            # 📖 БАЗОВІ ТЕРМІНИ
            "large language model": 40, "llm": 35, "language model": 30,
            "foundation model": 45, "generative model": 35,
            "pre-trained model": 35, "neural": 20,
            "deep learning": 25, "machine learning": 20,
            "artificial intelligence": 20, "natural language processing": 30,
            "nlp": 25, "text": 15,
        }
        
        # Авторитетні автори/установи (бонус)
        self.authority_bonus = {
            "openai": 30, "anthropic": 30, "google": 25, "deepmind": 30,
            "meta": 25, "microsoft": 20, "stanford": 25, "mit": 25,
            "berkeley": 25, "carnegie mellon": 25, "oxford": 20, "cambridge": 20,
            "yann lecun": 35, "geoffrey hinton": 35, "yoshua bengio": 35,
            "demis hassabis": 30, "ilya sutskever": 30, "andrej karpathy": 30,
            "sam altman": 25, "dario amodei": 30,
        }

    def _keyword_score(self, paper_data):
        """Оцінює статтю за ключовими словами."""
        text = (paper_data['title'] + " " + paper_data['summary']).lower()
        authors_str = ' '.join(paper_data['authors']).lower()
        score = 0
        
        # Рахуємо score з ключових слів
        for keyword, weight in self.keyword_scores.items():
            if keyword in text:
                score += weight
         
        # Бонус за авторитетних авторів/установи
        for authority, bonus in self.authority_bonus.items():
            if authority in authors_str:
                score += bonus
             
        return int(score)

    def _freshness_score(self, paper_data):
        """Додає бонус за свіжість статті."""
        pub_date = datetime.fromisoformat(paper_data['published_date'])
        days_old = (datetime.now().date() - pub_date.date()).days
        
        if days_old < 30:
            return 50
        elif days_old < 90:
            return 30
        elif days_old < 180:
            return 15
        elif days_old < 365:
            return 5
        return 0

    def _categorize_paper(self, paper_data):
        """Визначає категорії статті (розширено!)."""
        text = (paper_data['title'] + " " + paper_data['summary']).lower()
        
        categories = []
        
        # Перевірка категорій
        if any(word in text for word in ["survey", "review", "taxonomy", "systematic review", "overview"]):
            categories.append("📚 Survey")
        if any(word in text for word in ["agent", "autonomous", "multi-agent", "agentic", "tool-using"]):
            categories.append("🤖 Agents")
        if any(word in text for word in ["prompt", "in-context", "few-shot", "chain-of-thought", "cot", "react"]):
            categories.append("✍️ Prompting")
        if any(word in text for word in ["benchmark", "evaluation", "metric", "empirical", "leaderboard"]):
            categories.append("📊 Evaluation")
        if any(word in text for word in ["rag", "retrieval", "vector", "embedding"]):
            categories.append("🔍 RAG")
        if any(word in text for word in ["reasoning", "planning", "problem solving", "logical"]):
            categories.append("🧠 Reasoning")
        if any(word in text for word in ["safety", "injection", "jailbreak", "adversarial", "red teaming"]):
            categories.append("🔒 Safety")
        if any(word in text for word in ["context window", "long context", "extended context", "memory"]):
            categories.append("📏 Context")
        if any(word in text for word in ["gpt-4", "gpt-5", "claude", "gemini", "llama", "mistral", "new model"]):
            categories.append("🆕 New Models")
        if any(word in text for word in ["fine-tuning", "rlhf", "alignment", "instruction tuning"]):
            categories.append("🎓 Training")
        if any(word in text for word in ["multimodal", "vision", "image", "cross-modal"]):
            categories.append("🌍 Multimodal")
        if any(word in text for word in ["code", "programming", "program synthesis"]):
            categories.append("💻 Code")
        
        return " | ".join(categories) if categories else "📄 General"

    def _llm_score_paper(self, paper_data, citations):
        """Етап 3: LLM оцінка з розширеними критеріями."""
        try:
            abstract = paper_data['summary'][:800]
            authors_preview = ', '.join(paper_data['authors'][:5])
            
            prompt = f"""Rate this LLM research paper's importance (0-100).

Title: {paper_data['title']}
Authors: {authors_preview}
Abstract: {abstract}
Citations: {citations}

Key criteria:
- Survey/Review: +40-50
- Novel prompting techniques: +30-40
- New models or architectures: +35-45
- LLM capabilities analysis: +30-40
- Context engineering: +30-40
- AI agents: +30-40
- Evaluation methods: +25-35
- Top institutions: +20-30
- High citations: +20-30
- Practical applications: +20-30

Return format:
SCORE: [number]
REASON: [one sentence why this matters]"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert AI researcher evaluating paper importance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            
            try:
                score_line = [l for l in result.split('\n') if 'SCORE:' in l][0]
                reason_line = [l for l in result.split('\n') if 'REASON:' in l][0]
                
                score = int(score_line.split(':')[1].strip())
                reason = reason_line.split(':', 1)[1].strip()
                
                return score, reason
            except:
                return 50, "Evaluation format error"
                
        except Exception as e:
            return 50, f"LLM error: {str(e)[:50]}"

    def _get_citations(self, paper_data):
        """Отримує кількість цитувань через Semantic Scholar API."""
        try:
            arxiv_id = paper_data['entry_id'].split('/')[-1]
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
            params = {"fields": "citationCount"}
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('citationCount', 0)
            return 0
        except:
            return 0

    def generate_summary(self, abstract):
        """Генерує короткий саммарі через OpenAI."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Summarize this research paper abstract in 2-3 sentences, focusing on key contributions and practical implications."},
                    {"role": "user", "content": abstract}
                ],
                max_tokens=150,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return abstract[:200] + "..."

    def search_papers(self, start_date, end_date, max_results=300, 
                     min_keyword_score=40, top_n_for_citations=100, 
                     top_n_for_llm=50, final_top_n=30):
        """
        Гібридний пошук у 3 етапи
        """
        
        existing_ids = set()
        
        # ПРОСТИЙ запит - фільтрація через keywords у Stage 1
        query = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG"
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        print(f"\n{'='*60}")
        print("🚀 HYBRID SCRAPER v2.1 - РОЗШИРЕНА LLM ЕКОСИСТЕМА")
        print(f"{'='*60}\n")

        # ═══════════════════════════════════════════════════════════
        # ЕТАП 1: KEYWORD FILTERING
        # ═══════════════════════════════════════════════════════════
        print("📊 STAGE 1: Keyword-based filtering...")
        found_papers = []
        
        try:
            for result in tqdm(self.arxiv_client.results(search), desc="Searching papers", total=max_results):
                try:
                    published_date = result.published.date()
                    
                    if result.pdf_url in existing_ids:
                        continue
                    
                    if start_date.date() <= published_date <= end_date.date():
                        paper_data = {
                            'entry_id': result.entry_id,
                            'title': result.title,
                            'authors': [author.name for author in result.authors],
                            'summary': result.summary,
                            'published_date': published_date.isoformat(),
                            'pdf_url': result.pdf_url,
                            'primary_category': result.primary_category
                        }
                        
                        keyword_score = self._keyword_score(paper_data)
                        freshness_score = self._freshness_score(paper_data)
                        
                        paper_data['keyword_score'] = keyword_score
                        paper_data['freshness_score'] = freshness_score
                        paper_data['stage1_score'] = keyword_score + freshness_score
                        
                        if paper_data['stage1_score'] >= min_keyword_score:
                            found_papers.append(paper_data)
                            
                except Exception as paper_error:
                    # Skip individual paper errors
                    continue
                    
        except arxiv.UnexpectedEmptyPageError:
            print(f"\n⚠️ ArXiv returned empty page. Continuing with {len(found_papers)} papers found...")
        except Exception as e:
            print(f"\n⚠️ Search error: {e}. Continuing with {len(found_papers)} papers...")
        
        print(f"✅ Found {len(found_papers)} papers with keyword score >= {min_keyword_score}\n")
        
        if not found_papers:
            print("❌ No papers found!")
            return []
        
        found_papers.sort(key=lambda p: p['stage1_score'], reverse=True)
        
        # ═══════════════════════════════════════════════════════════
        # ЕТАП 2: CITATION SCORING
        # ═══════════════════════════════════════════════════════════
        print(f"📈 STAGE 2: Fetching citations for top {top_n_for_citations} papers...\n")
        top_for_citations = found_papers[:top_n_for_citations]
        
        for paper in tqdm(top_for_citations, desc="Fetching citations"):
            citations = self._get_citations(paper)
            paper['citations'] = citations
            
            citation_score = min(citations * 2, 100)
            paper['citation_score'] = citation_score
            paper['stage2_score'] = int(
                paper['stage1_score'] * 0.7 + citation_score * 0.3
            )
            time.sleep(0.1)
        
        for paper in found_papers[top_n_for_citations:]:
            paper['citations'] = 0
            paper['citation_score'] = 0
            paper['stage2_score'] = paper['stage1_score']
        
        found_papers.sort(key=lambda p: p['stage2_score'], reverse=True)
        print(f"✅ Citations fetched!\n")
        
        # ═══════════════════════════════════════════════════════════
        # ЕТАП 3: LLM EVALUATION
        # ═══════════════════════════════════════════════════════════
        print(f"🤖 STAGE 3: LLM evaluation for top {top_n_for_llm} papers...\n")
        top_for_llm = found_papers[:top_n_for_llm]
        
        for paper in tqdm(top_for_llm, desc="LLM scoring"):
            llm_score, llm_reason = self._llm_score_paper(paper, paper['citations'])
            paper['llm_score'] = llm_score
            paper['llm_reason'] = llm_reason
            paper['final_score'] = int(
                paper['stage1_score'] * 0.3 +
                paper['citation_score'] * 0.2 +
                llm_score * 0.5
            )
        
        for paper in found_papers[top_n_for_llm:]:
            paper['llm_score'] = 0
            paper['llm_reason'] = "Not evaluated"
            paper['final_score'] = int(paper['stage2_score'] * 0.6)
        
        found_papers.sort(key=lambda p: p['final_score'], reverse=True)
        
        for paper in found_papers:
            paper['categories'] = self._categorize_paper(paper)
        
        print(f"✅ LLM evaluation complete!\n")
        
        final_papers = found_papers[:final_top_n]
        
        print(f"\n{'='*60}")
        print(f"🎯 FINAL RESULTS: Top {len(final_papers)} papers")
        print(f"{'='*60}\n")
        
        print("📌 Top 5 Papers:\n")
        for i, paper in enumerate(final_papers[:5], 1):
            print(f"{i}. [{paper['final_score']}] {paper['title'][:60]}...")
            print(f"   {paper['categories']}")
            print(f"   📊 Keyword: {paper['keyword_score']} | 📈 Citations: {paper['citations']} | 🤖 LLM: {paper['llm_score']}")
            if paper.get('llm_reason'):
                print(f"   💭 {paper['llm_reason']}")
            print()
        
        return final_papers

    def save_to_csv(self, papers, filename):
        if not papers:
            return
        fieldnames = [
            'final_score', 'categories', 'title', 'citations', 
            'keyword_score', 'llm_score', 'llm_reason',
            'authors', 'published_date', 'pdf_url', 'primary_category',
            'stage1_score', 'stage2_score', 'freshness_score', 'citation_score'
        ]
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for paper in papers:
                    paper_copy = paper.copy()
                    if 'authors' in paper_copy and isinstance(paper_copy['authors'], list):
                        paper_copy['authors'] = ', '.join(paper_copy['authors'])
                    writer.writerow(paper_copy)
            print(f"💾 CSV saved to {filename}")
        except Exception as e:
            print(f"❌ Error: {e}")

    def save_to_json(self, papers, filename):
        if not papers:
            return
        try:
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(papers, jsonfile, ensure_ascii=False, indent=2)
            print(f"💾 JSON saved to {filename}")
        except Exception as e:
            print(f"❌ Error: {e}")

    def save_to_markdown(self, papers, filename, generate_summaries=False):
        if not papers:
            return
        try:
            with open(filename, 'w', encoding='utf-8') as mdfile:
                mdfile.write("# 🎯 Top ArXiv Papers: LLM Ecosystem\n\n")
                mdfile.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                mdfile.write(f"**Total papers:** {len(papers)}\n\n")
                mdfile.write("---\n\n")
                
                categories_count = {}
                for paper in papers:
                    cats = paper.get('categories', 'Unknown').split(' | ')
                    for cat in cats:
                        categories_count[cat] = categories_count.get(cat, 0) + 1
                
                mdfile.write("## 📊 Categories Distribution\n\n")
                for cat, count in sorted(categories_count.items(), key=lambda x: x[1], reverse=True):
                    mdfile.write(f"- {cat}: {count} papers\n")
                mdfile.write("\n---\n\n")
                
                mdfile.write("## 📚 Papers\n\n")
                
                for i, paper in enumerate(papers, 1):
                    authors_str = ', '.join(paper['authors'][:5]) if isinstance(paper['authors'], list) else paper['authors']
                    if len(paper.get('authors', [])) > 5:
                        authors_str += " et al."
                    
                    mdfile.write(f"### {i}. {paper['title']}\n\n")
                    mdfile.write(f"**🎯 Score:** {paper.get('final_score', 'N/A')} | ")
                    mdfile.write(f"**📈 Citations:** {paper.get('citations', 0)} | ")
                    mdfile.write(f"**📅 Published:** {paper['published_date']}\n\n")
                    mdfile.write(f"**Categories:** {paper.get('categories', 'N/A')}\n\n")
                    
                    mdfile.write(f"<details>\n<summary>📊 Scoring Details</summary>\n\n")
                    mdfile.write(f"- Keyword: {paper.get('keyword_score', 0)}\n")
                    mdfile.write(f"- Citation: {paper.get('citation_score', 0)}\n")
                    mdfile.write(f"- LLM: {paper.get('llm_score', 0)}\n")
                    mdfile.write(f"- Freshness: {paper.get('freshness_score', 0)}\n")
                    if paper.get('llm_reason'):
                        mdfile.write(f"\n**LLM Reasoning:** {paper['llm_reason']}\n")
                    mdfile.write(f"</details>\n\n")
                    
                    mdfile.write(f"**👥 Authors:** {authors_str}\n\n")
                    mdfile.write(f"**🔗 Links:** [ArXiv]({paper['pdf_url']}) | [PDF]({paper['pdf_url']}.pdf)\n\n")
                    
                    if generate_summaries:
                        if not paper.get('ai_summary'):
                            paper['ai_summary'] = self.generate_summary(paper['summary'])
                        mdfile.write(f"#### 💡 AI Summary\n\n> {paper['ai_summary']}\n\n")
                    
                    mdfile.write(f"<details>\n<summary>📄 Abstract</summary>\n\n{paper['summary']}\n\n</details>\n\n")
                    mdfile.write("---\n\n")
            
            print(f"💾 Markdown saved to {filename}")
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    from datetime import datetime, timedelta
    
    scraper = ArxivLLMScraperHybrid()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    print(f"\n🔍 Searching papers from {start_date.date()} to {end_date.date()}\n")
    
    papers = scraper.search_papers(
        start_date=start_date,
        end_date=end_date,
        max_results=300,
        min_keyword_score=40,
        top_n_for_citations=100,
        top_n_for_llm=50,
        final_top_n=30
    )
    
    if papers:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        scraper.save_to_csv(papers, f"papers_v21_{timestamp}.csv")
        scraper.save_to_json(papers, f"papers_v21_{timestamp}.json")
        scraper.save_to_markdown(papers, f"papers_v21_{timestamp}.md", generate_summaries=False)
        print(f"\n✅ Done! Version 2.1 with expanded LLM focus")
    else:
        print("\n❌ No papers found!")

if __name__ == "__main__":
    main()