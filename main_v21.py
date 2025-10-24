#!/usr/bin/env python3
"""
ArXiv LLM Research Toolkit v2.1
Автоматичний пошук та summarization AI/ML статей з arXiv
Підтримує v2.0 (фокус на агентах) та v2.1 (вся LLM екосистема)
"""

from datetime import datetime
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='🔬 ArXiv LLM Research Toolkit v2.1 - Search and summarize AI papers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:
  # Базовий запуск (v2.1, останні 6 місяців)
  python main_v21.py
  
  # Використати v2.0 (фокус на агентах)
  python main_v21.py --version 2.0
  
  # Кастомний період з LLM саммарі
  python main_v21.py --start-date 2025-01-01 --use-llm
  
  # Більше результатів
  python main_v21.py --max-results 500 --top-llm 100
        """
    )
    
    # === НОВИЙ ПАРАМЕТР: Вибір версії ===
    parser.add_argument('--version', type=str, default='2.1', choices=['2.0', '2.1'],
                       help='Scraper version: 2.0 (agents focus) or 2.1 (full LLM ecosystem, default)')
    
    # Параметри дат
    parser.add_argument('--start-date', type=str, 
                       default=(datetime.now().replace(month=datetime.now().month-6 if datetime.now().month > 6 else datetime.now().month+6, 
                                                       year=datetime.now().year if datetime.now().month > 6 else datetime.now().year-1)).strftime('%Y-%m-%d'),
                       help='Start date in YYYY-MM-DD format (default: 6 months ago)')
    parser.add_argument('--end-date', type=str, 
                       default=datetime.now().strftime('%Y-%m-%d'), 
                       help='End date in YYYY-MM-DD format (default: today)')
    
    # Параметри пошуку
    parser.add_argument('--max-results', type=int, default=300, 
                       help='Maximum papers to fetch from ArXiv (default: 300)')
    parser.add_argument('--min-score', type=int, default=20,
                       help='Minimum keyword score to pass Stage 1 (default: 40)')
    parser.add_argument('--top-citations', type=int, default=100,
                       help='Top N papers to get citations for (Stage 2, default: 100)')
    parser.add_argument('--top-llm', type=int, default=50,
                       help='Top N papers to evaluate with LLM (Stage 3, default: 50)')
    parser.add_argument('--final-top', type=int, default=30,
                       help='Final number of papers to save (default: 30)')
    
    # LLM опції
    parser.add_argument('--use-llm', action='store_true', 
                       help='Enable LLM evaluation (Stage 3)')
    parser.add_argument('--no-llm', action='store_true',
                       help='Disable LLM evaluation (budget mode, $0)')
    parser.add_argument('--generate-summaries', action='store_true',
                       help='Generate AI summaries for all papers (costs extra)')
    
    # Вихідні файли
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--output-prefix', type=str, default='papers',
                       help='Prefix for output files (default: papers)')
    
    args = parser.parse_args()
    
    # === Імпорт правильної версії ===
    if args.version == '2.1':
        try:
            from arxiv_llm_scraper_hybrid_v21 import ArxivLLMScraperHybrid
            version_name = "v2.1 (Full LLM Ecosystem)"
        except ImportError:
            print("⚠️ v2.1 not found, falling back to v2.0")
            from arxiv_llm_scraper_hybrid import ArxivLLMScraperHybrid
            version_name = "v2.0 (Agents Focus)"
    else:
        from arxiv_llm_scraper_hybrid import ArxivLLMScraperHybrid
        version_name = "v2.0 (Agents Focus)"
    
    # Створюємо output директорію
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Генеруємо імена файлів з timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    version_suffix = args.version.replace('.', '')
    json_path = os.path.join(args.output_dir, f"{args.output_prefix}_v{version_suffix}_{timestamp}.json")
    csv_path = os.path.join(args.output_dir, f"{args.output_prefix}_v{version_suffix}_{timestamp}.csv")
    md_path = os.path.join(args.output_dir, f"{args.output_prefix}_v{version_suffix}_{timestamp}.md")
    
    # Конвертація дат
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"❌ Invalid date format: {e}")
        print("Use YYYY-MM-DD format (e.g., 2025-01-01)")
        return
    
    # Визначаємо чи використовувати LLM
    use_llm = args.use_llm or not args.no_llm  # За замовчуванням LLM enabled
    if args.no_llm:
        args.top_llm = 0  # Budget mode
    
    # === Красивий вивід параметрів ===
    print("\n" + "=" * 70)
    print("🚀 ArXiv LLM Research Toolkit v2.1")
    print("=" * 70)
    print(f"📌 Version:          {version_name}")
    print(f"📅 Period:           {args.start_date} to {args.end_date}")
    print(f"🔍 Max results:      {args.max_results} papers")
    print(f"📊 Min keyword score: {args.min_score}")
    print()
    print("🎯 Pipeline Configuration:")
    print(f"   Stage 1 (Keywords):  All papers → filter by score >= {args.min_score}")
    print(f"   Stage 2 (Citations): Top {args.top_citations} papers")
    print(f"   Stage 3 (LLM):       {'Top ' + str(args.top_llm) + ' papers' if args.top_llm > 0 else 'DISABLED (Budget Mode)'}")
    print(f"   Final output:        Top {args.final_top} papers")
    print()
    print(f"🤖 LLM Features:")
    print(f"   Evaluation:      {'✅ Enabled' if use_llm and args.top_llm > 0 else '❌ Disabled'}")
    print(f"   AI Summaries:    {'✅ Enabled' if args.generate_summaries else '❌ Disabled'}")
    print()
    print(f"💰 Estimated cost:   ${estimate_cost(args.top_llm, args.generate_summaries, args.final_top)}")
    print("=" * 70 + "\n")
    
    # === Пошук статей ===
    scraper = ArxivLLMScraperHybrid()
    
    try:
        papers = scraper.search_papers(
            start_date=start_date,
            end_date=end_date,
            max_results=args.max_results,
            min_keyword_score=args.min_score,
            top_n_for_citations=args.top_citations,
            top_n_for_llm=args.top_llm,
            final_top_n=args.final_top
        )
    except Exception as e:
        print(f"\n❌ Error during search: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # === Збереження результатів ===
    if papers:
        print(f"\n{'='*70}")
        print(f"💾 Saving results...")
        print(f"{'='*70}\n")
        
        try:
            scraper.save_to_json(papers, json_path)
            scraper.save_to_csv(papers, csv_path)
            scraper.save_to_markdown(papers, md_path, generate_summaries=args.generate_summaries)
            
            print(f"\n{'='*70}")
            print("✅ SUCCESS! Papers saved to:")
            print(f"{'='*70}")
            print(f"  📄 JSON:     {json_path}")
            print(f"  📊 CSV:      {csv_path}")
            print(f"  📖 Markdown: {md_path}")
            print(f"{'='*70}\n")
            
            # === Статистика ===
            print_statistics(papers, args.version)
            
        except Exception as e:
            print(f"\n❌ Error saving files: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️ No papers found matching the criteria.")
        print("Try:")
        print("  - Lowering --min-score (default: 40)")
        print("  - Increasing --max-results (default: 300)")
        print("  - Expanding date range")


def estimate_cost(top_llm, generate_summaries, final_top):
    """Оцінка вартості OpenAI API"""
    if top_llm == 0:
        return "0.00 (Budget Mode)"
    
    cost = 0.0
    
    # LLM evaluation (~100 tokens per paper)
    if top_llm > 0:
        cost += (top_llm * 100 * 0.00000015)  # GPT-4o-mini input
        cost += (top_llm * 50 * 0.0000006)    # GPT-4o-mini output
    
    # AI summaries (~150 tokens per paper)
    if generate_summaries:
        cost += (final_top * 800 * 0.00000015)  # input (abstract)
        cost += (final_top * 150 * 0.0000006)   # output (summary)
    
    return f"{cost:.4f}"


def print_statistics(papers, version):
    """Виводить статистику по знайдених статтях"""
    print("📊 STATISTICS")
    print("=" * 70)
    print(f"Total papers found:  {len(papers)}")
    
    # Статистика по категоріях
    categories_count = {}
    for paper in papers:
        cats = paper.get('categories', 'Unknown').split(' | ')
        for cat in cats:
            categories_count[cat] = categories_count.get(cat, 0) + 1
    
    print(f"\nTop Categories:")
    for cat, count in sorted(categories_count.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {cat}: {count} papers")
    
    # Статистика по scores
    avg_keyword = sum(p.get('keyword_score', 0) for p in papers) / len(papers) if papers else 0
    avg_citations = sum(p.get('citations', 0) for p in papers) / len(papers) if papers else 0
    avg_llm = sum(p.get('llm_score', 0) for p in papers if p.get('llm_score', 0) > 0)
    llm_count = len([p for p in papers if p.get('llm_score', 0) > 0])
    avg_llm = avg_llm / llm_count if llm_count > 0 else 0
    
    print(f"\nAverage Scores:")
    print(f"  Keyword Score:  {avg_keyword:.1f}")
    print(f"  Citations:      {avg_citations:.1f}")
    if avg_llm > 0:
        print(f"  LLM Score:      {avg_llm:.1f}")
    
    # Топ статті
    print(f"\n🏆 Top 3 Papers:")
    for i, paper in enumerate(papers[:3], 1):
        print(f"\n  {i}. [{paper.get('final_score', 0)}] {paper['title'][:65]}...")
        print(f"     {paper.get('categories', 'N/A')}")
        if paper.get('llm_reason'):
            print(f"     💭 {paper['llm_reason'][:60]}...")
    
    print("\n" + "=" * 70)


def show_version_comparison():
    """Показує порівняння версій"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║              VERSION COMPARISON: v2.0 vs v2.1                  ║
╚════════════════════════════════════════════════════════════════╝

v2.0 - Agents Focus:
  ✓ 70 keywords
  ✓ Strong focus on AI agents
  ✓ 7 categories
  ✓ Good for agent-specific research

v2.1 - Full LLM Ecosystem (RECOMMENDED):
  ✓ 150+ keywords (+114%)
  ✓ Balanced focus on entire LLM ecosystem
  ✓ 12 categories (+71%)
  ✓ Covers: prompting, new models, context, capabilities
  ✓ 2-3x more relevant papers found

Recommendation: Use v2.1 for most cases!
Use v2.0 only if you need ONLY agent papers.
    """)


if __name__ == "__main__":
    # Додаємо спеціальні команди
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        show_version_comparison()
        sys.exit(0)
    
    main()
