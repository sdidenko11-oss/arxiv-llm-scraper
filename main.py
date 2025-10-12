#!/usr/bin/env python3
"""
ArXiv LLM Research Toolkit
Автоматичний пошук та summarization AI/ML статей з arXiv
"""

from arxiv_llm_scraper import ArxivLLMScraper
from datetime import datetime
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='🔬 ArXiv LLM Research Toolkit - Search and summarize AI papers',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # --- Аргументи залишаються без змін ---
    parser.add_argument('--start-date', type=str, default='2025-09-25', help='Start date in YYYY-MM-DD format (default: 2025-09-25)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--max-results', type=int, default=200, help='Maximum number of results to fetch (default: 200)')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for summarization (requires OpenAI API key)')
    parser.add_argument('--output-json', type=str, default='results.json', help='JSON output filename (default: results.json)')
    parser.add_argument('--output-csv', type=str, default='results.csv', help='CSV output filename (default: results.csv)')
    parser.add_argument('--output-md', type=str, default='report.md', help='Markdown report filename (default: report.md)')
    
    args = parser.parse_args()
    
    # --- Логіка збереження файлів у папку 'results' ---

    # Визначаємо назву папки для результатів
    output_dir = "results"
    
    # Створюємо папку, якщо вона не існує. exist_ok=True запобігає помилці, якщо папка вже є.
    os.makedirs(output_dir, exist_ok=True)

    # Формуємо повні шляхи до файлів, додаючи ім'я папки
    json_path = os.path.join(output_dir, args.output_json)
    csv_path = os.path.join(output_dir, args.output_csv)
    md_path = os.path.join(output_dir, args.output_md)
    
    # Конвертація дат
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    print("=" * 60)
    print("🤖 ArXiv LLM Research Toolkit")
    print("=" * 60)
    print(f"📅 Period: {args.start_date} to {args.end_date}")
    print(f"📊 Max results: {args.max_results}")
    print(f"🤖 LLM Summarization: {'Enabled' if args.use_llm else 'Disabled'}")
    print("=" * 60)
    
    # Завантаження існуючих статей
    old_papers = []
    existing_ids = set()
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                old_papers = json.load(f)
                existing_ids = {paper['pdf_url'] for paper in old_papers}
            print(f"📖 Loaded {len(old_papers)} previously saved papers.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Could not read {json_path}. Starting fresh. Error: {e}")
    
    # Пошук нових статей
    scraper = ArxivLLMScraper()
    new_papers = scraper.search_papers(
        start_date=start_date,
        end_date=end_date,
        max_results=args.max_results,
        existing_ids=existing_ids,
        use_llm=args.use_llm
    )
    
    # Збереження результатів
    if new_papers:
        all_papers = old_papers + new_papers
        
        print(f"\n💾 Total papers in database: {len(all_papers)}. Updating files...")
        
        scraper.save_to_csv(all_papers, csv_path)
        scraper.save_to_json(all_papers, json_path)
        scraper.save_to_markdown(all_papers, md_path)
        
        print(f"\n✅ Successfully added {len(new_papers)} new papers!")
        
        # Статистика
        if args.use_llm:
            print(f"🤖 AI summaries generated: {len([p for p in new_papers if p.get('ai_summary')])}")
    else:
        print("\n✅ No new papers found for the specified period.")
    
    # Вивід фінальної інформації
    print("\n" + "=" * 60)
    print("📁 Output files:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()