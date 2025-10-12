#!/bin/bash

# ArXiv LLM Toolkit - Quick Start Script

echo "🤖 ArXiv LLM Research Toolkit"
echo "=============================="
echo ""

# Перевірка віртуального середовища
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Активація venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Встановлення залежностей
if [ ! -f "venv/installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/installed
fi

# Запуск
echo "🚀 Starting search..."
echo ""

# Варіанти запуску:
# 1. Без LLM (швидко, безкоштовно)
python3 main.py --start-date 2025-09-25 --max-results 100

# 2. З LLM (потребує API key, коштує ~$0.01-0.10)
# python main.py --start-date 2025-09-25 --max-results 100 --use-llm

echo ""
echo "✅ Done! Check report.md for results"
