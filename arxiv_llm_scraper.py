# arxiv_llm_scraper.py

from openai import OpenAI
import os
import csv
import json
import arxiv
from dotenv import load_dotenv
from tqdm import tqdm

class ArxivLLMScraper:
    def __init__(self):
        load_dotenv()
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def generate_summary(self, abstract, use_llm=True):
        # ... (цей метод без змін)
        if not use_llm:
            return abstract[:200] + "..."
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a research paper summarizer. Create concise 2-3 sentence summaries."},
                    {"role": "user", "content": f"Summarize this research abstract in 2-3 sentences:\n\n{abstract}"}
                ],
                max_tokens=150,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ LLM summarization failed: {e}")
            return abstract[:200] + "..."
    
    def search_papers(self, start_date, end_date, max_results=200, existing_ids=None, use_llm=True):
        # ... (цей метод без змін)
        if existing_ids is None:
            existing_ids = set()

        query = "cat:cs.CL AND (LLM OR \"Large Language Model\") AND (\"prompt engineering\" OR prompting OR \"in-context learning\") AND (evaluation OR benchmark OR metric OR performance OR assessment OR method OR principle OR pattern OR technique OR approach)"
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        newly_found_papers = []
        print("Searching and filtering papers...") 
        
        for result in tqdm(search.results(), desc="Searching papers", total=max_results):
            published_date = result.published.date()
            
            if result.pdf_url in existing_ids:
                continue

            if start_date.date() <= published_date <= end_date.date():
                ai_summary = self.generate_summary(result.summary, use_llm=use_llm) if use_llm else None
                
                paper_data = {
                    'entry_id': result.entry_id,
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary,
                    'ai_summary': ai_summary,
                    'published_date': published_date.isoformat(),
                    'pdf_url': result.pdf_url,
                    'primary_category': result.primary_category
                }
                newly_found_papers.append(paper_data)
                
                if len(newly_found_papers) % 5 == 0:
                    print(f"  ✓ Processed {len(newly_found_papers)} new papers...") 
        
        print(f"Filtering complete. Found {len(newly_found_papers)} NEW papers in the specified date range.") 
        return newly_found_papers

    def save_to_csv(self, papers, filename):
        # ... (цей метод без змін)
        if not papers:
            print("No papers to save to CSV.")
            return

        fieldnames = papers[0].keys()

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for paper in papers:
                    paper_copy = paper.copy()
                    if 'authors' in paper_copy and isinstance(paper_copy['authors'], list):
                        paper_copy['authors'] = ', '.join(paper_copy['authors'])
                    writer.writerow(paper_copy)
            print(f"💾 Data successfully saved to {filename}") 
        except Exception as e:
            print(f"❌ Error saving to CSV: {e}") 

    def save_to_json(self, papers, filename):
        # ... (цей метод без змін)
        if not papers:
            print("No papers to save to JSON.")
            return

        try:
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(papers, jsonfile, ensure_ascii=False, indent=4)
            print(f"💾 Data successfully saved to {filename}") 
        except Exception as e:
            print(f"❌ Error saving to JSON: {e}")

    # --- 👇👇👇 ОСЬ ФІНАЛЬНА ЗМІНА: ДОДАНО МЕТОД ДЛЯ MARKDOWN 👇👇👇 ---
    def save_to_markdown(self, papers, filename):
        """Зберігає список статей у красивий Markdown-файл."""
        if not papers:
            print("No papers to save to Markdown.") 
            return

        try:
            with open(filename, 'w', encoding='utf-8') as mdfile:
                for i, paper in enumerate(papers):
                    # Перетворюємо список авторів на рядок
                    authors_str = ', '.join(paper['authors']) if isinstance(paper['authors'], list) else paper['authors']
                    
                    mdfile.write(f"## {i+1}. [{paper['title']}]({paper['pdf_url']})\n\n")
                    mdfile.write(f"**Authors:** {authors_str}\n\n")
                    mdfile.write(f"**Published Date:** {paper['published_date']}\n\n")
                    
                    if paper.get('ai_summary'):
                        mdfile.write(f"### AI Summary\n\n")
                        mdfile.write(f"{paper['ai_summary']}\n\n")
                    
                    mdfile.write(f"### Abstract\n\n")
                    mdfile.write(f"{paper['summary']}\n\n")
                    mdfile.write("---\n\n")
            
            print(f"💾 Data successfully saved to {filename}") 

        except Exception as e:
            print(f"❌ Error saving to Markdown: {e}") 