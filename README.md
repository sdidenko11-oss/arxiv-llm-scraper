# 🤖 ArXiv LLM Research Toolkit

This tool automates the search, filtering, and summarization of scientific papers from arXiv related to Large Language Models (LLM) and prompt engineering.

## 🚀 Key Features

*   **Automated Search** with an expanded list of keywords including methods, patterns, and evaluation techniques.
*   **Date Range Filtering** to search for papers within a specific period.
*   **Duplicate Avoidance** by loading previously found papers and searching only for new ones.
*   **Optional AI Summarization** of abstracts using OpenAI GPT-4o-mini for quick analysis.
*   **Multiple Output Formats**: Saves results in `JSON`, `CSV`, and `Markdown` (.md) into a dedicated `results/` folder.
*   **Simple Configuration** and execution with a single script.

## ⚙️ Installation and Setup

#### 1. Clone the Repository
```bash
git clone <your-repository-link>
cd arxivs_llm_scraper
2. Configure the API Key
To use the AI summarization feature, you need to set up your OpenAI API key.
Create a file named .env in the project's root directory.
Add the following line to it, replacing sk-... with your actual key:
code
Code
OPENAI_API_KEY="sk-..."
▶️ How to Run
The easiest way to run the script is by using run.sh.
Make the script executable (only needs to be done once):
code
Bash
chmod +x run.sh
```2.  **Run the script:**
```bash
./run.sh
Operating Modes
You can easily switch between modes by editing the run.sh file:
Fast Mode (No AI Summarization):
By default, the script runs quickly without using OpenAI. This line is active:
code
Bash
python3 main.py --start-date 2025-09-25 --max-results 50
AI Summarization Mode:
To enable summary generation, comment out the line above (add a # at the beginning) and uncomment the line with the --use-llm flag (remove the #):
code
Bash
# python3 main.py --start-date 2025-09-25 --max-results 50
python3 main.py --start-date 2025-09-25 --max-results 50 --use-llm
You can also change the date (--start-date) and the number of results (--max-results) directly in the run.sh file.
📁 Output Files
All results are saved into the results/ directory:
results.json: Complete paper data in JSON format, suitable for further processing.
results.csv: Tabular data in CSV format, which can be opened in Excel or Google Sheets.
report.md: A human-readable report in Markdown format with links to PDFs and AI summaries (if the option was enabled).
