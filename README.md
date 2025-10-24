# 🔬 ArXiv LLM Research Toolkit v2.1

> Автоматичний пошук та аналіз найважливіших AI/LLM статей з ArXiv

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.1-green.svg)](https://github.com/yourusername/arxiv-llm-scraper)

---

## 🎯 Що це?

Інтелектуальний скрапер для пошуку та ранжування наукових статей про Large Language Models (LLM) з ArXiv. Використовує **3-етапну систему фільтрації**:

1. 📊 **Keyword Scoring** - 150+ ключових термінів
2. 📈 **Citation Analysis** - інтеграція з Semantic Scholar
3. 🤖 **LLM Evaluation** - оцінка важливості через GPT-4

**Версія 2.1** охоплює **ВСЮ LLM екосистему**: промптинг, нові моделі, context engineering, агенти, та багато іншого!

---

## ✨ Що нового у v2.1?

| Параметр | v2.0 | v2.1 | Покращення |
|----------|------|------|------------|
| **Ключових термінів** | 70 | **150+** | +114% 🚀 |
| **Категорій** | 7 | **12** | +71% |
| **Фокус на промптинг** | Середній | **Високий** | +50% |
| **Нові моделі** | Базовий | **Розширений** | +300% |
| **Context engineering** | ❌ | **✅** | НОВЕ! |
| **Знайдених статей** | 100% | **200-300%** | 2-3x більше |

### 🆕 Нові категорії в v2.1:
- ✍️ Prompting (усі техніки: CoT, ReAct, ToT)
- 🆕 New Models (GPT-5, Claude 4, Gemini 2, Llama 4)
- 📏 Context Engineering (long context, memory, 1M tokens)
- 🧠 LLM Capabilities (reasoning, code, multilingual)
- 🎓 Training (RLHF, DPO, LoRA, alignment)
- 🌍 Multimodal (vision-language models)

---

## 🚀 Швидкий старт (3 команди!)

```bash
# 1. Клонуй репозиторій
git clone https://github.com/yourusername/arxiv-llm-scraper.git
cd arxiv-llm-scraper

# 2. Створи .env файл з API ключем
echo "OPENAI_API_KEY=your-key-here" > .env

# 3. Запусти!
chmod +x run.sh
./run.sh
```

**Вибери режим** з інтерактивного меню і отримай результати у папці `results/`! 🎉

---

## 📦 Встановлення

### Автоматичне (рекомендовано):

```bash
./run.sh
```

Скрипт автоматично:
- ✅ Створить віртуальне середовище
- ✅ Встановить залежності
- ✅ Запустить пошук

### Ручне:

```bash
# Створи віртуальне середовище
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# або
venv\Scripts\activate.bat  # Windows

# Встанови залежності
pip install -r requirements.txt

# Запусти
python3 main_v21.py
```

---

## 🎮 Режими роботи

### Через `run.sh` (найпростіше):

```bash
./run.sh
```

**Доступні режими:**

| # | Режим | Опис | Вартість | Час |
|---|-------|------|----------|-----|
| 1 | Quick Test | Останній місяць, без LLM | $0 | ~2 хв |
| 2 | Balanced | 6 місяців, з LLM (50 papers) | ~$0.01 | ~5 хв |
| 3 | Budget | 6 місяців, без LLM | $0 | ~3 хв |
| 4 | Premium | 6 місяців, LLM + summaries | ~$0.05 | ~10 хв |
| 5 | Agents Only | Використовує v2.0 (тільки агенти) | ~$0.01 | ~5 хв |
| 6 | Compare | Порівняти v2.0 vs v2.1 | $0 | ~4 хв |

### Через Python скрипт:

```bash
# Базовий запуск
python3 main_v21.py

# З параметрами
python3 main_v21.py \
    --start-date 2025-01-01 \
    --max-results 500 \
    --top-llm 100 \
    --final-top 50 \
    --generate-summaries

# Без LLM (безкоштовно)
python3 main_v21.py --no-llm

# Використати v2.0 замість v2.1
python3 main_v21.py --version 2.0
```

### Через Python код:

```python
from arxiv_llm_scraper_hybrid_v21 import ArxivLLMScraperHybrid
from datetime import datetime, timedelta

scraper = ArxivLLMScraperHybrid()

papers = scraper.search_papers(
    start_date=datetime.now() - timedelta(days=180),
    end_date=datetime.now(),
    max_results=300,
    min_keyword_score=40,
    top_n_for_citations=100,
    top_n_for_llm=50,
    final_top_n=30
)

scraper.save_to_json(papers, 'results.json')
scraper.save_to_csv(papers, 'results.csv')
scraper.save_to_markdown(papers, 'results.md')
```

---

## 📊 Параметри

### Основні:

| Параметр | За замовчуванням | Опис |
|----------|------------------|------|
| `--version` | 2.1 | Версія скрапера (2.0 або 2.1) |
| `--start-date` | 6 місяців тому | Дата початку (YYYY-MM-DD) |
| `--end-date` | Сьогодні | Дата кінця |
| `--max-results` | 300 | Максимум статей з ArXiv |

### Pipeline:

| Параметр | За замовчуванням | Опис |
|----------|------------------|------|
| `--min-score` | 40 | Мінімальний keyword score (Stage 1) |
| `--top-citations` | 100 | Скільки оцінювати цитуваннями (Stage 2) |
| `--top-llm` | 50 | Скільки оцінювати LLM (Stage 3) |
| `--final-top` | 30 | Кінцева кількість статей |

### LLM:

| Параметр | Опис |
|----------|------|
| `--use-llm` | Увімкнути LLM оцінку |
| `--no-llm` | Вимкнути LLM (budget mode, $0) |
| `--generate-summaries` | Генерувати AI саммарі для всіх статей |

---

## 🏗️ Архітектура (3-етапний pipeline)

```
ArXiv API (400 papers)
        ↓
┌───────────────────────────────────────┐
│  STAGE 1: Keyword Filtering           │
│  150+ weighted keywords               │
│  Freshness bonus                      │
│  Authority bonus                      │
└───────────────────────────────────────┘
        ↓ (filtered: ~150 papers)
┌───────────────────────────────────────┐
│  STAGE 2: Citation Analysis           │
│  Semantic Scholar API                 │
│  Citation count → score               │
└───────────────────────────────────────┘
        ↓ (top 100 by citations)
┌───────────────────────────────────────┐
│  STAGE 3: LLM Evaluation              │
│  GPT-4o-mini assessment               │
│  Weighted scoring (30/20/50)          │
└───────────────────────────────────────┘
        ↓ (top 30 papers)
    📄 Results
```

---

## 📂 Структура проекту

```
arxiv-llm-scraper/
├── 🎮 run.sh                           # Головний скрипт запуску (Linux/Mac)
├── 🪟 run.bat                          # Скрипт запуску (Windows)
├── ✨ arxiv_llm_scraper_hybrid_v21.py  # Скрапер v2.1 (ОСНОВНИЙ)
├── 📌 arxiv_llm_scraper_hybrid.py      # Скрапер v2.0 (для порівняння)
├── 🎯 main_v21.py                      # CLI інтерфейс
├── 🆚 compare_versions.py              # Порівняння v2.0 vs v2.1
├── 📦 requirements.txt                 # Залежності
├── 📚 README.md                        # Цей файл
├── 🇺🇦 README.uk.md                    # Українська версія
├── 📖 RUN_SCRIPTS_README.md            # Інструкції для run.sh
├── 🔒 .env                             # API ключі (створи сам)
├── 🗂️ results/                         # Результати пошуку
│   ├── papers_v21_*.json
│   ├── papers_v21_*.csv
│   └── papers_v21_*.md
└── 🐍 venv/                            # Віртуальне середовище
```

---

## 🔑 Налаштування

### 1. OpenAI API Key (обов'язково для LLM режимів)

Створи файл `.env` у корені проекту:

```bash
OPENAI_API_KEY=sk-proj-your-key-here
```

Отримати ключ: https://platform.openai.com/api-keys

### 2. Semantic Scholar API (опціонально)

Працює без API ключа, але з обмеженням rate limit.

---

## 💰 Вартість використання

| Режим | Papers | LLM Calls | Summaries | Вартість |
|-------|--------|-----------|-----------|----------|
| **Budget** | 300 | 0 | ❌ | **$0.00** |
| **Quick Test** | 100 | 0 | ❌ | **$0.00** |
| **Balanced** | 300 | 50 | ❌ | **~$0.01** |
| **Premium** | 400 | 100 | ✅ | **~$0.05** |

**Примітка:** Використовується GPT-4o-mini (~$0.00015 за 1K токенів)

---

## 📊 Приклади виводу

### Console:

```
🚀 HYBRID SCRAPER v2.1 - РОЗШИРЕНА LLM ЕКОСИСТЕМА
============================================================

📊 STAGE 1: Keyword-based filtering...
Searching papers: 100%|████████████████| 300/300 [00:45<00:00, 6.67it/s]
✅ Found 147 papers with keyword score >= 40

📈 STAGE 2: Fetching citations for top 100 papers...
Fetching citations: 100%|████████████| 100/100 [00:12<00:00, 8.15it/s]
✅ Citations fetched!

🤖 STAGE 3: LLM evaluation for top 50 papers...
LLM scoring: 100%|██████████████████| 50/50 [01:23<00:00, 1.67s/it]
✅ LLM evaluation complete!

🎯 FINAL RESULTS: Top 30 papers
============================================================

📌 Top 5 Papers:

1. [187] LLM Agents: A Survey of Autonomous AI Systems
   🤖 Agents | 📚 Survey
   📊 Keyword: 95 | 📈 Citations: 42 | 🤖 LLM: 92
   💭 Comprehensive survey covering agent architectures and applications

2. [174] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
   ✍️ Prompting | 🧠 Reasoning
   📊 Keyword: 85 | 📈 Citations: 156 | 🤖 LLM: 89
   💭 Breakthrough paper introducing CoT technique

...
```

### Markdown файл:

Повний звіт з:
- 📊 Розподіл по категоріях
- 📚 Детальна інформація про кожну статтю
- 🔗 Посилання на ArXiv та PDF
- 📄 Абстракти (згорнуті)
- 💡 AI саммарі (опціонально)
- 📈 Детальні оцінки

---

## 🎓 Використання

### Сценарій 1: Щоденний моніторинг

```bash
# Щодня запускай Budget режим
./run.sh
# Вибери: 3 (Budget Mode)

# Результати у results/papers_v21_*.md
```

### Сценарій 2: Глибоке дослідження

```bash
# Раз на тиждень Premium режим
./run.sh
# Вибери: 4 (Premium Mode)

# Отримаєш:
# - 50 найкращих статей
# - LLM оцінки
# - AI саммарі
```

### Сценарій 3: Автоматизація

```bash
# Додай у crontab (щодня о 9:00)
0 9 * * * cd /home/user/arxiv-llm-scraper && ./run.sh <<< "3"

# Або через systemd timer
```

---

## 🔧 Розширені можливості

### Порівняння версій v2.0 vs v2.1

```bash
python3 compare_versions.py
```

Покаже:
- Скільки статей знайшла кожна версія
- Унікальні статті для кожної версії
- Різниця в покритті категорій

### Кастомні параметри

```bash
# Тільки свіжі статті (останній місяць)
python3 main_v21.py --start-date 2025-09-24

# Велика вибірка
python3 main_v21.py --max-results 1000 --final-top 100

# Низький поріг (більше статей)
python3 main_v21.py --min-score 20

# Без цитувань (швидше)
python3 main_v21.py --top-citations 0 --top-llm 0
```

---

## 🐛 Troubleshooting

### Проблема: ArXiv повертає порожню сторінку

```
UnexpectedEmptyPageError
```

**Рішення:** Зменш `--max-results` або використовуй Budget/Balanced режими.

### Проблема: OpenAI API error

```
OpenAI API key not found
```

**Рішення:** Створи `.env` файл з `OPENAI_API_KEY=...`

### Проблема: Повільна робота

**Рішення:** 
- Використовуй `--no-llm` для пришвидшення
- Зменш `--max-results`
- Зменш `--top-llm`

### Проблема: Не знаходить статті

**Рішення:**
- Знижуй `--min-score` (з 40 до 20-30)
- Збільшуй `--max-results`
- Розшируй діапазон дат

---

## 📖 Документація

- 📚 [README.md](README.md) - Цей файл
- 🇺🇦 [README.uk.md](README.uk.md) - Українська версія
- 🚀 [RUN_SCRIPTS_README.md](RUN_SCRIPTS_README.md) - Інструкції для run.sh
- 📋 [CHANGELOG_v21.md](CHANGELOG_v21.md) - Історія змін (якщо є)

---

## 🤝 Contributing

Pull requests welcome! Для великих змін спочатку відкрийте issue.

**Ідеї для contribution:**
- 📧 Email нотифікації
- 📱 Telegram бот
- 🌐 Web інтерфейс
- 📊 Візуалізація трендів
- 🔄 Інкрементальне оновлення
- 🧠 ML для оптимізації ваг

---

## 📜 License

MIT License - використовуй вільно!

---

## 🙏 Acknowledgments

- [ArXiv](https://arxiv.org/) - за відкритий доступ до наукових статей
- [Semantic Scholar](https://www.semanticscholar.org/) - за API для цитувань
- [OpenAI](https://openai.com/) - за GPT API

---

## 📞 Контакти

- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/arxiv-llm-scraper/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/arxiv-llm-scraper/discussions)

---

## ⭐ Star History

Якщо проект корисний - постав ⭐ на GitHub!

---

**Створено з ❤️ для LLM researchers**

*Last updated: October 2025*
