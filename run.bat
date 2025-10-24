@echo off
REM ArXiv LLM Toolkit v2.1 - Windows Quick Start
REM Автоматичний запуск для Windows

echo ========================================
echo   ArXiv LLM Research Toolkit v2.1
echo ========================================
echo.

REM === Перевірка Python ===
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Install Python 3.8+ first.
    pause
    exit /b 1
)

REM === Перевірка віртуального середовища ===
if not exist "venv\" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
)

REM === Активація venv ===
echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat

REM === Встановлення залежностей ===
if not exist "venv\installed" (
    echo [SETUP] Installing dependencies...
    
    if exist "requirements.txt" (
        pip install -r requirements.txt
    ) else (
        echo [WARN] requirements.txt not found. Installing manually...
        pip install openai arxiv python-dotenv requests tqdm
    )
    
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    
    echo. > venv\installed
    echo [OK] Dependencies installed successfully
)

REM === Перевірка .env ===
if not exist ".env" (
    echo.
    echo [WARN] .env file not found!
    echo        LLM features will be disabled.
    echo        Create .env with: OPENAI_API_KEY=your-key-here
    echo.
)

REM === Меню вибору ===
echo.
echo ========================================
echo   Choose run mode:
echo ========================================
echo.
echo 1. Quick Test (last month, no LLM, fast)
echo 2. Balanced Mode (6 months, with LLM, ~$0.01) [RECOMMENDED]
echo 3. Budget Mode (6 months, no LLM, $0)
echo 4. Premium Mode (6 months, LLM+summaries, ~$0.05)
echo 5. Agents Only (v2.0, agent focus)
echo 6. Compare Versions (v2.0 vs v2.1)
echo 7. Exit
echo.
set /p choice="Enter choice [1-7]: "

echo.
echo [RUN] Starting search...
echo.

if "%choice%"=="1" (
    echo [MODE] Quick Test
    python main_v21.py --max-results 100 --no-llm --final-top 10
    
) else if "%choice%"=="2" (
    echo [MODE] Balanced ^(RECOMMENDED^)
    python main_v21.py --max-results 300 --top-llm 50 --final-top 30
    
) else if "%choice%"=="3" (
    echo [MODE] Budget
    python main_v21.py --max-results 300 --no-llm --final-top 30
    
) else if "%choice%"=="4" (
    echo [MODE] Premium
    python main_v21.py --max-results 400 --top-llm 100 --final-top 50 --generate-summaries
    
) else if "%choice%"=="5" (
    echo [MODE] Agents Only ^(v2.0^)
    python main_v21.py --version 2.0 --max-results 300 --top-llm 50 --final-top 30
    
) else if "%choice%"=="6" (
    echo [MODE] Compare Versions
    python compare_versions.py
    
) else if "%choice%"=="7" (
    echo Exiting...
    exit /b 0
    
) else (
    echo [ERROR] Invalid choice
    pause
    exit /b 1
)

REM === Результати ===
if errorlevel 1 (
    echo.
    echo ========================================
    echo [ERROR] Something went wrong
    echo ========================================
    pause
    exit /b 1
) else (
    echo.
    echo ========================================
    echo [OK] Done! Results saved to:
    echo ========================================
    echo.
    echo   results\
    echo     - papers_*.json
    echo     - papers_*.csv
    echo     - papers_*.md
    echo.
    echo Open the .md file for best experience!
    echo.
    pause
)
