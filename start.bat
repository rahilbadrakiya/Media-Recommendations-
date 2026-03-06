@echo off
echo.
echo  =========================================
echo   CineMate Ultra — Starting App
echo  =========================================
echo.

echo [1/2] Starting FastAPI backend on http://localhost:8000
start cmd /k "cd backend && python main.py"
timeout /t 2 /nobreak > nul

echo [2/2] Starting Vite frontend on http://localhost:5173
start cmd /k "cd frontend && npm run dev"

echo.
echo  Open http://localhost:5173 in your browser
echo  AI Chatbot: click the "AI Pick" bubble (bottom-right)
echo.
