@echo off
setlocal enabledelayedexpansion

echo ===================================
echo Universal Agent Chain Setup
echo ===================================

:: Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Install required packages
echo.
echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install required packages
    pause
    exit /b 1
)

:: Create .env file
echo.
echo Setting up OpenAI API key...
set /p OPENAI_API_KEY="Please enter your OpenAI API key: "
echo OPENAI_API_KEY=%OPENAI_API_KEY%> .env
echo API key saved to .env file

:: Get the current directory
set "CURRENT_DIR=%CD%"

:: Create desktop shortcuts
echo.
echo Creating desktop shortcuts...

:: Create shortcut for console version
set "SHORTCUT_PATH_CONSOLE=%USERPROFILE%\Desktop\LegalCaseProcessor.lnk"
set "TARGET_PATH_CONSOLE=%CURRENT_DIR%\run_legal_case.bat"

:: Create the run_legal_case.bat
echo @echo off > run_legal_case.bat
echo cd /d "%CURRENT_DIR%" >> run_legal_case.bat
echo set PYTHONPATH=%CURRENT_DIR% >> run_legal_case.bat
echo python run_legal_case.py >> run_legal_case.bat
echo pause >> run_legal_case.bat

:: Create shortcut for Streamlit version
set "SHORTCUT_PATH_STREAMLIT=%USERPROFILE%\Desktop\AgentChainBuilder.lnk"
set "TARGET_PATH_STREAMLIT=%CURRENT_DIR%\run_streamlit.bat"

:: Create the run_streamlit.bat
echo @echo off > run_streamlit.bat
echo cd /d "%CURRENT_DIR%" >> run_streamlit.bat
echo set PYTHONPATH=%CURRENT_DIR% >> run_streamlit.bat
echo streamlit run agent_chain_app.py >> run_streamlit.bat
echo pause >> run_streamlit.bat

:: Create the shortcuts using PowerShell
powershell -Command "$WS = New-Object -ComObject WScript.Shell; $SC = $WS.CreateShortcut('%SHORTCUT_PATH_CONSOLE%'); $SC.TargetPath = '%TARGET_PATH_CONSOLE%'; $SC.WorkingDirectory = '%CURRENT_DIR%'; $SC.Save()"
powershell -Command "$WS = New-Object -ComObject WScript.Shell; $SC = $WS.CreateShortcut('%SHORTCUT_PATH_STREAMLIT%'); $SC.TargetPath = '%TARGET_PATH_STREAMLIT%'; $SC.WorkingDirectory = '%CURRENT_DIR%'; $SC.Save()"

echo.
echo ===================================
echo Setup completed successfully!
echo.
echo Two desktop shortcuts have been created:
echo 1. 'LegalCaseProcessor' - Console Version
echo    - Run the legal case processor in console mode
echo.
echo 2. 'AgentChainBuilder' - Streamlit Version
echo    - Run the universal agent chain builder with web interface
echo.
echo You can start either version by:
echo - Double-clicking the respective desktop shortcut
echo - Running run_legal_case.bat or run_streamlit.bat directly
echo ===================================
echo.
pause 