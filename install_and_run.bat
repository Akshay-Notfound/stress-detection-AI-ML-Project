@echo off
echo Installing stress detection dependencies...
echo ======================================

REM Create virtual environment
echo Creating virtual environment...
python -m venv stress_detection_env
if %errorlevel% neq 0 (
    echo Failed to create virtual environment
    exit /b %errorlevel%
)

REM Activate virtual environment
echo Activating virtual environment...
call stress_detection_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment
    exit /b %errorlevel%
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip
    exit /b %errorlevel%
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies
    exit /b %errorlevel%
)

echo.
echo Installation completed successfully!
echo.
echo To run the Streamlit app, execute:
echo   streamlit run app/streamlit_app.py
echo.
echo To run the test pipeline, execute:
echo   python test_pipeline.py
echo.
echo To run the Jupyter notebooks, execute:
echo   jupyter notebook
echo.
echo Press any key to exit...
pause >nul