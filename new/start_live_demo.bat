@echo off
echo 🚀 Starting Brain Tumor Classification Live Demo...
echo.
echo 📍 Opening demo at http://localhost:8501
echo.
echo ⚠️  Press Ctrl+C to stop the demo
echo.

cd /d "C:\Users\tommi\OneDrive\Desktop\Main_Project\new"
C:\Users\tommi\OneDrive\Desktop\Main_Project\new\.venv\Scripts\streamlit.exe run live_demo.py --server.port 8501 --server.headless false

pause