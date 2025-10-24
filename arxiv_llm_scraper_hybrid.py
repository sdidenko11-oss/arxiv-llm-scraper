# arxiv_llm_scraper_hybrid.py

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
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Розширені ключові слова з фокусом на Agents, Prompting та LLM
        self.keyword_scores = {
            # Оглядові статті (найвища цінність)
            "survey": 100, "review": 100, "taxonomy": 90, "systematic review": 90,
            "comprehensive": 70, "state-of-the-art": 80,
            
            # AI Агенти (дуже важливо!)
            "autonomous agents": 60, "ai agent": 55, "multi-agent": 50, "multiagent": 50,
            "agent framework": 45, "agentic": 40, "tool-using": 40, "tool use": 40,
            "langchain": 35, "autogpt": 40, "babyagi": 40, "agent-based": 45,
            "planning": 35, "reasoning agent": 50, "cognitive architecture": 45,
            
            # Prompt Engineering техніки (ключова тема)
            "chain-of-thought": 50, "cot": 45, "self-consistency": 45,
            "tree of thoughts": 45, "tot": 40, "graph of thoughts": 40,
            "react": 45, "reason and act": 45, "reflexion": 40, "reflection": 35,
            "least-to-most": 35, "self-ask": 35, "maieutic": 30,
            "automatic prompt": 50, "prompt optimization": 45, "prompt tuning": 40,
            "meta-prompting": 45, "prompt pattern": 40,
            
            # In-Context Learning
            "in-context learning": 45, "icl": 40, "few-shot": 35, "zero-shot": 35,
            "one-shot": 30, "demonstration": 30,
            
            # Evaluation & Benchmarks
            "benchmark": 40, "evaluation": 35, "metric": 30, "assessment": 30,
            "human evaluation": 45, "llm-as-judge": 40, "llm as a judge": 40,
            "automated evaluation": 35,
            
            # Практичні техніки
            "retrieval-augmented": 40, "rag": 40, "retrieval augmented generation": 40,
            "instruction tuning": 40, "instruction following": 35,
            "rlhf": 35, "reinforcement learning from human feedback": 40,
            "constitutional ai": 40, "alignment": 30,
            "json mode": 25, "structured output": 30, "function calling": 35,
            "tool calling": 35, "api": 20,
            
            # Безпека та надійність
            "prompt injection": 35, "jailbreak": 30, "adversarial": 30,
            "robustness": 30, "safety": 25, "hallucination": 35,
            "factuality": 35, "truthfulness": 30,
            
            # Загальні LLM терміни (менша вага)
            "large language model": 10, "llm": 10, "language model": 10,
            "gpt": 15, "claude": 15, "palm": 15, "llama": 15,
        }
        
        self.authority_bonus = {
            # Топові AI лабораторії
            "openai": 30, "anthropic": 30, "google": 25, "deepmind": 25, "google deepmind": 30,
            "meta": 20, "meta ai": 20, "microsoft": 20, "microsoft research": 25,
            "apple": 15, "nvidia": 15, "cohere": 15,
            
            # Топові університети
            "stanford": 20, "mit": 20, "berkeley": 20, "cmu": 20, "carnegie mellon": 20,
            "princeton": 15, "oxford": 15, "cambridge": 15, "harvard": 15,
            "toronto": 15, "montreal": 15, "eth zurich": 15, "tsinghua": 15,
            
            # Відомі дослідники (можна розширити)
            "yann lecun": 25, "andrew ng": 25, "yoshua bengio": 25,
            "geoffrey hinton": 25, "ilya sutskever": 25, "demis hassabis": 25,
        }

    @staticmethod
    def _get_citation_count(arxiv_id):
        """Отримує кількість цитувань з Semantic Scholar API."""
        try:
            time.sleep(0.5)  # Збільшена затримка для стабільності
            api_url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}?fields=citationCount"
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("citationCount", 0)
        except requests.exceptions.RequestException as e:
            # print(f"⚠️ Citation fetch failed for {arxiv_id}: {e}")
            return 0

    def _score_paper_keywords(self, paper_data):
        """Базова оцінка за ключовими словами та авторитетом."""
        score = 0
        title = paper_data['title'].lower()
        summary = paper_data['summary'].lower()
        authors_str = ', '.join(paper_data['authors']).lower()
        
        text_content = title + " " + summary

        # 1. Оцінка за ключовими словами
        for keyword, value in self.keyword_scores.items():
            if keyword in text_content:
                score += value
                # Бонус якщо ключове слово в заголовку
                if keyword in title:
                    score += value * 0.3
        
        # 2. Бонус за авторитетних авторів/установи
        for authority, bonus in self.authority_bonus.items():
            if authority in authors_str:
                score += bonus
        
        return int(score)

    def _add_recency_bonus(self, paper_data, base_score):
        """Додає бонус за свіжість статті."""
        pub_date = datetime.fromisoformat(paper_data['published_date'])
        days_old = (datetime.now().date() - pub_date.date()).days
        
        recency_bonus = 0
        if days_old < 30:
            recency_bonus = 50  # Дуже свіжа!
        elif days_old < 90:
            recency_bonus = 30
        elif days_old < 180:
            recency_bonus = 15
        elif days_old < 365:
            recency_bonus = 5
        
        return base_score + recency_bonus

    def _categorize_paper(self, paper_data):
        """Визначає категорії статті."""
        text = (paper_data['title'] + " " + paper_data['summary']).lower()
        
        categories = []
        
        if any(word in text for word in ["survey", "review", "taxonomy", "systematic"]):
            categories.append("📚 Survey")
        if any(word in text for word in ["agent", "autonomous", "multi-agent", "agentic", "tool use", "tool-using"]):
            categories.append("🤖 Agents")
        if any(word in text for word in ["prompt", "in-context", "few-shot", "chain-of-thought", "cot"]):
            categories.append("✍️ Prompting")
        if any(word in text for word in ["benchmark", "evaluation", "metric", "assessment"]):
            categories.append("📊 Evaluation")
        if any(word in text for word in ["rag", "retrieval"]):
            categories.append("🔍 RAG")
        if any(word in text for word in ["hallucination", "factuality", "truthfulness", "safety"]):
            categories.append("🛡️ Safety")
        
        return " | ".join(categories) if categories else "📄 General"

    def _llm_score_paper(self, paper_data, citations):
        """Використовує LLM для детальної оцінки важливості статті."""
        try:
            authors_preview = ', '.join(paper_data['authors'][:5])
            if len(paper_data['authors']) > 5:
                authors_preview += f" (and {len(paper_data['authors']) - 5} more)"
            
            prompt = f"""Rate this paper's importance for prompt engineering, LLM agents, and AI research (0-100).

Title: {paper_data['title']}

Authors: {authors_preview}

Abstract (first 600 chars): {paper_data['summary'][:600]}...

Citations: {citations}

Scoring criteria:
- Survey/review papers: very high value (+40-50)
- Novel prompting techniques (CoT, ToT, ReAct, etc.): high value (+30-40)
- AI agents, autonomous systems, multi-agent: high value (+30-40)
- Benchmarks and evaluation methods: medium-high (+25-35)
- Papers from OpenAI/Anthropic/Google/DeepMind: bonus (+15-25)
- High citation count: bonus (citations * 1.0)
- Practical implementation: medium (+20-30)

Reply with ONLY a number between 0-100. No explanation."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            score_text = response.choices[0].message.content.strip()
            return int(score_text)
        
        except Exception as e:
            print(f"⚠️ LLM scoring failed: {e}")
            return 0

    def generate_summary(self, abstract, use_llm=True):
        """Генерує стислий саммарі статті."""
        if not use_llm:
            return abstract[:200] + "..."
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a research paper summarizer. Create concise 2-3 sentence summaries focusing on the key contribution and practical implications."},
                    {"role": "user", "content": f"Summarize this research abstract in 2-3 sentences:\n\n{abstract}"}
                ],
                max_tokens=150,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ LLM summarization failed: {e}")
            return abstract[:200] + "..."

    def search_papers(self, start_date, end_date, max_results=300, 
                     min_keyword_score=30, top_n_for_llm=100, 
                     final_top_n=50, existing_ids=None, use_llm_scoring=True):
        """
        Гібридний пошук та ранжування статей.
        
        Параметри:
        - max_results: максимум статей для пошуку
        - min_keyword_score: мінімальний score за ключовими словами
        - top_n_for_llm: скільки топових статей оцінювати через LLM
        - final_top_n: скільки фінальних статей повернути
        - use_llm_scoring: чи використовувати LLM для оцінки
        """
        if existing_ids is None:
            existing_ids = set()

        # Покращений запит з фокусом на агентів та промпти
        query = """
        (cat:cs.CL OR cat:cs.AI OR cat:cs.LG) AND 
        (
            ("prompt engineering" OR prompting OR "in-context learning" OR "few-shot" OR "chain-of-thought") OR
            ("language model agent" OR "autonomous agent" OR "multi-agent" OR "agentic" OR "tool use") OR
            ("llm benchmark" OR "language model evaluation")
        )
        """
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        found_papers = []
        print(f"🔍 Searching ArXiv papers (max {max_results})...")
        
        # Етап 1: Збираємо статті в діапазоні дат
        for result in tqdm(search.results(), desc="Collecting papers", total=max_results):
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
                found_papers.append(paper_data)
        
        print(f"✅ Found {len(found_papers)} new papers in date range")
        
        if not found_papers:
            print("❌ No papers found. Try adjusting date range or query.")
            return []

        # Етап 2: Базова оцінка за ключовими словами + авторитет
        print(f"\n📊 Stage 1: Scoring by keywords and authority...")
        papers_with_keyword_scores = []
        
        for paper in tqdm(found_papers, desc="Keyword scoring"):
            keyword_score = self._score_paper_keywords(paper)
            
            if keyword_score >= min_keyword_score:
                # Додаємо бонус за свіжість
                keyword_score = self._add_recency_bonus(paper, keyword_score)
                paper['keyword_score'] = keyword_score
                paper['category'] = self._categorize_paper(paper)
                papers_with_keyword_scores.append(paper)
        
        print(f"✅ {len(papers_with_keyword_scores)} papers passed keyword threshold (>= {min_keyword_score})")
        
        if not papers_with_keyword_scores:
            print("❌ No papers met minimum keyword score. Try lowering min_keyword_score.")
            return []
        
        # Сортуємо за keyword score
        papers_with_keyword_scores.sort(key=lambda p: p['keyword_score'], reverse=True)
        
        # Етап 3: Отримуємо цитування для топових статей
        print(f"\n📚 Stage 2: Fetching citations for top {top_n_for_llm} papers...")
        top_papers_for_citation = papers_with_keyword_scores[:top_n_for_llm]
        
        for paper in tqdm(top_papers_for_citation, desc="Getting citations"):
            arxiv_id = paper['entry_id'].split('/')[-1].replace('v1', '').replace('v2', '').replace('v3', '')
            paper['citations'] = self._get_citation_count(arxiv_id)
        
        # Етап 4: LLM оцінка (опціонально, але рекомендовано)
        if use_llm_scoring:
            print(f"\n🤖 Stage 3: LLM scoring top {top_n_for_llm} papers...")
            
            for paper in tqdm(top_papers_for_citation, desc="LLM evaluation"):
                llm_score = self._llm_score_paper(paper, paper.get('citations', 0))
                
                # Комбінований score: 60% LLM + 40% keyword
                paper['llm_score'] = llm_score
                paper['final_score'] = int(llm_score * 0.6 + paper['keyword_score'] * 0.4)
                paper['citation_boost'] = int(paper.get('citations', 0) * 1.5)
                paper['final_score'] += paper['citation_boost']
            
            # Сортуємо за фінальним score
            top_papers_for_citation.sort(key=lambda p: p['final_score'], reverse=True)
        else:
            # Якщо без LLM - просто використовуємо keyword score + citations
            for paper in top_papers_for_citation:
                paper['citation_boost'] = int(paper.get('citations', 0) * 1.5)
                paper['final_score'] = paper['keyword_score'] + paper['citation_boost']
                paper['llm_score'] = 0  # Не використовували
            
            top_papers_for_citation.sort(key=lambda p: p['final_score'], reverse=True)
        
        # Етап 5: Фінальна вибірка
        final_papers = top_papers_for_citation[:final_top_n]
        
        print(f"\n✨ Stage 4: Generating AI summaries for top {len(final_papers)} papers...")
        for paper in tqdm(final_papers, desc="Generating summaries"):
            paper['ai_summary'] = self.generate_summary(paper['summary'])
        
        # Додаємо ранг
        for i, paper in enumerate(final_papers):
            paper['rank'] = i + 1
        
        print(f"\n🎯 Final Result: {len(final_papers)} top papers selected!")
        print(f"   Score range: {final_papers[0]['final_score']} - {final_papers[-1]['final_score']}")
        
        return final_papers

    def save_to_csv(self, papers, filename):
        """Зберігає статті у CSV з усіма метриками."""
        if not papers:
            print("No papers to save to CSV.")
            return

        fieldnames = ['rank', 'final_score', 'llm_score', 'keyword_score', 'citations', 
                     'citation_boost', 'category', 'title', 'authors', 'published_date', 
                     'pdf_url', 'ai_summary', 'summary', 'primary_category']

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for paper in papers:
                    paper_copy = paper.copy()
                    if 'authors' in paper_copy and isinstance(paper_copy['authors'], list):
                        paper_copy['authors'] = '; '.join(paper_copy['authors'])
                    writer.writerow(paper_copy)
            print(f"💾 Data successfully saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving to CSV: {e}")

    def save_to_json(self, papers, filename):
        """Зберігає статті у JSON."""
        if not papers:
            print("No papers to save to JSON.")
            return

        try:
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(papers, jsonfile, ensure_ascii=False, indent=2)
            print(f"💾 Data successfully saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving to JSON: {e}")

    def save_to_markdown(self, papers, filename):
        """Зберігає статті у красивий Markdown з повною інформацією."""
        if not papers:
            print("No papers to save to Markdown.")
            return

        try:
            with open(filename, 'w', encoding='utf-8') as mdfile:
                mdfile.write("# 🌟 Top ArXiv Papers: Prompt Engineering & LLM Agents\n\n")
                mdfile.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
                mdfile.write(f"**Total papers analyzed: {len(papers)}**\n\n")
                mdfile.write("---\n\n")
                
                for paper in papers:
                    authors_str = ', '.join(paper['authors'][:3])
                    if len(paper['authors']) > 3:
                        authors_str += f" et al. ({len(paper['authors'])} authors)"
                    
                    mdfile.write(f"## {paper['rank']}. {paper['title']}\n\n")
                    
                    # Метрики
                    mdfile.write(f"**📊 Scores:** Final: `{paper.get('final_score', 0)}` | ")
                    mdfile.write(f"LLM: `{paper.get('llm_score', 0)}` | ")
                    mdfile.write(f"Keywords: `{paper.get('keyword_score', 0)}` | ")
                    mdfile.write(f"Citations: `{paper.get('citations', 0)}` ⭐\n\n")
                    
                    mdfile.write(f"**🏷️ Category:** {paper.get('category', 'N/A')}\n\n")
                    mdfile.write(f"**👥 Authors:** {authors_str}\n\n")
                    mdfile.write(f"**📅 Published:** {paper['published_date']}\n\n")
                    mdfile.write(f"**🔗 PDF:** [{paper['pdf_url']}]({paper['pdf_url']})\n\n")
                    
                    if paper.get('ai_summary'):
                        mdfile.write(f"### 💡 AI Summary\n\n")
                        mdfile.write(f"> {paper['ai_summary']}\n\n")
                    
                    mdfile.write(f"### 📝 Abstract\n\n")
                    mdfile.write(f"{paper['summary']}\n\n")
                    mdfile.write("---\n\n")
            
            print(f"💾 Data successfully saved to {filename}")

        except Exception as e:
            print(f"❌ Error saving to Markdown: {e}")

    def print_top_papers_summary(self, papers, top_n=10):
        """Виводить короткий саммарі топових статей в консоль."""
        print(f"\n{'='*80}")
        print(f"📋 TOP {min(top_n, len(papers))} PAPERS SUMMARY")
        print(f"{'='*80}\n")
        
        for paper in papers[:top_n]:
            print(f"#{paper['rank']} | Score: {paper['final_score']} | 📚 {paper.get('citations', 0)} citations")
            print(f"   {paper.get('category', '')}")
            print(f"   📄 {paper['title']}")
            print(f"   🔗 {paper['pdf_url']}")
            print()
