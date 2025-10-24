# 🚀 Скрипти запуску - Quick Start

## 📋 Що є:

1. **`run.sh`** - для Linux/Mac
2. **`run.bat`** - для Windows
3. **`requirements.txt`** - залежності

---

## 🐧 Linux/Mac

### Перший запуск:

```bash
# Зроби файл виконуваним
chmod +x run.sh

# Запусти
./run.sh
```

### Наступні запуски:

```bash
./run.sh
```

### Ручний запуск:

```bash
# Створи venv
python3 -m venv venv

# Активуй
source venv/bin/activate

# Встанови залежності
pip install -r requirements.txt

# Запусти
python3 main_v21.py
```

---

## 🪟 Windows

### Перший запуск:

Подвійний клік на **`run.bat`**

Або з командного рядка:
```cmd
run.bat
```

### Наступні запуски:

Просто запусти **`run.bat`** знову.

### Ручний запуск:

```cmd
REM Створи venv
python -m venv venv

REM Активуй
venv\Scripts\activate.bat

REM Встанови залежності
pip install -r requirements.txt

REM Запусти
python main_v21.py
```

---

## 🎯 Режими запуску

Обидва скрипти (`run.sh` та `run.bat`) мають однакові режими:

### 1️⃣ Quick Test
- Останній місяць
- Без LLM (швидко)
- 10 статей
- **Вартість: $0**
- **Час: ~2 хв**

### 2️⃣ Balanced Mode ⭐ (РЕКОМЕНДОВАНО)
- Останні 6 місяців
- З LLM (50 статей)
- 30 статей
- **Вартість: ~$0.01**
- **Час: ~4 хв**

### 3️⃣ Budget Mode
- Останні 6 місяців
- Без LLM
- 30 статей
- **Вартість: $0**
- **Час: ~2 хв**

### 4️⃣ Premium Mode
- Останні 6 місяців
- З LLM (100 статей)
- AI summaries
- 50 статей
- **Вартість: ~$0.05**
- **Час: ~8 хв**

### 5️⃣ Agents Only (v2.0)
- Використовує стару версію 2.0
- Фокус на AI агентах
- **Вартість: ~$0.01**

### 6️⃣ Compare Versions
- Порівнює v2.0 vs v2.1
- Показує різницю
- **Вартість: $0**

### 7️⃣ Custom (тільки в run.sh)
- Власні параметри
- Повний контроль

---

## ⚙️ Що роблять скрипти?

1. ✅ Перевіряють Python
2. ✅ Створюють віртуальне середовище (venv)
3. ✅ Встановлюють залежності
4. ✅ Перевіряють .env файл
5. ✅ Запускають обраний режим
6. ✅ Показують результати

---

## 📝 .env файл

**ВАЖЛИВО!** Для режимів з LLM потрібен файл `.env`:

```bash
# Створи файл .env
echo "OPENAI_API_KEY=your-key-here" > .env
```

Або створи вручну:

**.env:**
```
OPENAI_API_KEY=sk-proj-abc123xyz...
```

Без `.env` файлу:
- ✅ Працюють режими без LLM (Budget, Quick Test без LLM)
- ❌ НЕ працюють режими з LLM

---

## 🐛 Troubleshooting

### Проблема: "Python not found"
```bash
# Linux/Mac
sudo apt install python3 python3-venv  # Ubuntu/Debian
brew install python3                   # Mac

# Windows
Завантаж з python.org
```

### Проблема: "Permission denied" (Linux/Mac)
```bash
chmod +x run.sh
```

### Проблема: "pip install fails"
```bash
# Спробуй
pip install --user -r requirements.txt

# Або
python3 -m pip install -r requirements.txt
```

### Проблема: "Module not found"
```bash
# Переконайся що venv активований
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate.bat # Windows

# Встанови ще раз
pip install -r requirements.txt
```

### Проблема: "API key error"
```bash
# Перевір .env файл
cat .env  # Linux/Mac
type .env # Windows

# Має бути:
# OPENAI_API_KEY=sk-proj-...
```

---

## 💡 Поради

### Рекомендований workflow:

1. **Перший раз:** Quick Test
   ```bash
   ./run.sh
   # Вибери опцію 1
   ```

2. **Якщо все ОК:** Balanced Mode
   ```bash
   ./run.sh
   # Вибери опцію 2
   ```

3. **Для економії:** Budget Mode
   ```bash
   ./run.sh
   # Вибери опцію 3
   ```

### Автоматизація (Linux/Mac):

```bash
# Щоденний запуск через cron
crontab -e

# Додай рядок (щодня о 9:00):
0 9 * * * cd /path/to/project && ./run.sh <<< "3"
```

### Автоматизація (Windows):

1. Відкрий Task Scheduler
2. Create Basic Task
3. Тригер: Daily
4. Дія: Start a program
5. Program: `C:\path\to\project\run.bat`

---

## 📊 Порівняння методів запуску

| Метод | Переваги | Недоліки |
|-------|----------|----------|
| **run.sh/bat** | ✅ Просто<br>✅ Меню<br>✅ Автоматично все | ❌ Менше контролю |
| **main_v21.py** | ✅ Повний контроль<br>✅ Всі параметри | ❌ Треба знати команди |
| **Python код** | ✅ Гнучкість<br>✅ Інтеграція | ❌ Треба писати код |

**Рекомендація:** Почни з `run.sh`/`run.bat`, потім перейди на `main_v21.py` для точніших налаштувань.

---

## 🎯 Швидкі команди

### Linux/Mac:
```bash
# Перший запуск
chmod +x run.sh && ./run.sh

# Balanced mode (автоматично)
echo "2" | ./run.sh

# Budget mode (автоматично)
echo "3" | ./run.sh
```

### Windows:
```cmd
REM Просто запусти
run.bat

REM Або подвійний клік
```

---

## ✅ Готово!

Тепер у тебе є **прості скрипти** для запуску ArXiv scraper!

**Наступні кроки:**
1. Запусти `./run.sh` (Linux/Mac) або `run.bat` (Windows)
2. Вибери режим (рекомендую #2 - Balanced)
3. Дивись результати у папці `results/`

**Удачі! 🚀**
