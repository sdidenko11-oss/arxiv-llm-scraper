#!/bin/bash

# ArXiv LLM Toolkit v2.1 - Quick Start Script
# Автоматичний запуск з підтримкою v2.0 та v2.1

echo "🤖 ArXiv LLM Research Toolkit v2.1"
echo "======================================"
echo ""

# === Перевірка віртуального середовища ===
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create venv. Try: python3 -m pip install --user virtualenv"
        exit 1
    fi
fi

# === Активація venv ===
echo "🔧 Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate venv"
    exit 1
fi

# === Встановлення залежностей ===
if [ ! -f "venv/installed" ]; then
    echo "📥 Installing dependencies..."
    
    # Перевірка наявності requirements.txt
    if [ ! -f "requirements.txt" ]; then
        echo "⚠️  requirements.txt not found. Installing manually..."
        pip install openai arxiv python-dotenv requests tqdm
    else
        pip install -r requirements.txt
    fi
    
    if [ $? -eq 0 ]; then
        touch venv/installed
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

# === Перевірка .env файлу ===
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  WARNING: .env file not found!"
    echo "   LLM features will be disabled unless you create .env with:"
    echo "   OPENAI_API_KEY=your-key-here"
    echo ""
    echo "   Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 0
    fi
fi

# === Вибір режиму запуску ===
echo ""
echo "🎯 Choose run mode:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1) Quick Test (v2.1, last month, no LLM, fast)"
echo "2) Balanced Mode (v2.1, 6 months, with LLM, ~\$0.01)"
echo "3) Budget Mode (v2.1, 6 months, no LLM, \$0)"
echo "4) Premium Mode (v2.1, 6 months, LLM+summaries, ~\$0.05)"
echo "5) Agents Only (v2.0, 6 months, agent focus)"
echo "6) Compare Versions (test v2.0 vs v2.1)"
echo "7) Custom (enter your own parameters)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -n "Enter choice [1-7]: "
read -r choice

echo ""
echo "🚀 Starting search..."
echo ""

case $choice in
    1)
        # Quick Test - швидка перевірка
        echo "📊 Mode: Quick Test"
        echo "   • Version: v2.1"
        echo "   • Period: Last 30 days"
        echo "   • LLM: Disabled"
        echo "   • Cost: \$0"
        echo ""
        python3 main_v21.py \
            --start-date $(date -d '30 days ago' +%Y-%m-%d) \
            --max-results 100 \
            --no-llm \
            --final-top 10
        ;;
    
    2)
        # Balanced Mode - рекомендований
        echo "📊 Mode: Balanced (RECOMMENDED)"
        echo "   • Version: v2.1"
        echo "   • Period: Last 6 months"
        echo "   • LLM: Enabled (50 papers)"
        echo "   • Cost: ~\$0.01"
        echo ""
        python3 main_v21.py \
            --start-date $(date -d '6 months ago' +%Y-%m-%d) \
            --max-results 300 \
            --top-llm 50 \
            --final-top 30
        ;;
    
    3)
        # Budget Mode - безкоштовно
        echo "📊 Mode: Budget"
        echo "   • Version: v2.1"
        echo "   • Period: Last 6 months"
        echo "   • LLM: Disabled"
        echo "   • Cost: \$0"
        echo ""
        python3 main_v21.py \
            --start-date $(date -d '6 months ago' +%Y-%m-%d) \
            --max-results 300 \
            --no-llm \
            --final-top 30
        ;;
    
    4)
        # Premium Mode - максимальна якість
        echo "📊 Mode: Premium"
        echo "   • Version: v2.1"
        echo "   • Period: Last 6 months"
        echo "   • LLM: Enabled (100 papers)"
        echo "   • AI Summaries: Yes"
        echo "   • Cost: ~\$0.05"
        echo ""
        python3 main_v21.py \
            --start-date $(date -d '6 months ago' +%Y-%m-%d) \
            --max-results 400 \
            --top-llm 100 \
            --final-top 50 \
            --generate-summaries
        ;;
    
    5)
        # Agents Only (v2.0)
        echo "📊 Mode: Agents Only (v2.0)"
        echo "   • Version: v2.0"
        echo "   • Period: Last 6 months"
        echo "   • Focus: AI Agents"
        echo "   • Cost: ~\$0.01"
        echo ""
        python3 main_v21.py \
            --version 2.0 \
            --start-date $(date -d '6 months ago' +%Y-%m-%d) \
            --max-results 300 \
            --top-llm 50 \
            --final-top 30
        ;;
    
    6)
        # Compare Versions
        echo "📊 Mode: Compare v2.0 vs v2.1"
        echo "   • Testing both versions"
        echo "   • Period: Last 2 months"
        echo "   • LLM: Disabled (for speed)"
        echo ""
        python3 compare_versions.py
        ;;
    
    7)
        # Custom - власні параметри
        echo "📊 Mode: Custom"
        echo ""
        echo "Enter parameters (or press Enter for defaults):"
        echo ""
        
        echo -n "Version [2.1]: "
        read -r version
        version=${version:-2.1}
        
        echo -n "Start date (YYYY-MM-DD) [6 months ago]: "
        read -r start_date
        start_date=${start_date:-$(date -d '6 months ago' +%Y-%m-%d)}
        
        echo -n "Max results [300]: "
        read -r max_results
        max_results=${max_results:-300}
        
        echo -n "Use LLM? (y/n) [y]: "
        read -r use_llm
        use_llm=${use_llm:-y}
        
        if [[ "$use_llm" =~ ^[Yy]$ ]]; then
            echo -n "Top N for LLM [50]: "
            read -r top_llm
            top_llm=${top_llm:-50}
            llm_flag="--top-llm $top_llm"
        else
            llm_flag="--no-llm"
        fi
        
        echo -n "Final top N [30]: "
        read -r final_top
        final_top=${final_top:-30}
        
        echo ""
        echo "Running with:"
        echo "  Version: $version"
        echo "  Start date: $start_date"
        echo "  Max results: $max_results"
        echo "  LLM: $use_llm"
        echo "  Final top: $final_top"
        echo ""
        
        python3 main_v21.py \
            --version $version \
            --start-date $start_date \
            --max-results $max_results \
            $llm_flag \
            --final-top $final_top
        ;;
    
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# === Результати ===
if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Done! Results saved to:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📂 results/"
    echo "   ├── 📄 papers_*.json"
    echo "   ├── 📊 papers_*.csv"
    echo "   └── 📖 papers_*.md"
    echo ""
    echo "💡 Tip: Open the .md file for best reading experience!"
    echo ""
else
    echo ""
    echo "❌ Error occurred. Check the output above."
    exit 1
fi