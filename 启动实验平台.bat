@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo 电动工具实验平台 — 公开轴承数据与虚拟实验台
if exist ".venv\Scripts\python.exe" goto checkdeps
py -3.12 -m venv .venv
if errorlevel 1 (
  echo 请先安装 Python 3.12，并勾选安装 Python Launcher，再重新打开此文件。
  pause
  exit /b 1
)
:checkdeps
.venv\Scripts\python.exe -c "import streamlit,torch,scipy,pandas,numpy"
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install -r requirements.lock.txt
  if errorlevel 1 (
    echo 依赖安装失败，请保留上面的报错信息。
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe scripts\verify_phase2.py
if errorlevel 1 (
  echo 公开数据或模型校验失败，请重新下载完整项目 ZIP。
  pause
  exit /b 1
)
echo 启动后在浏览器输入 http://localhost:8501
 echo 使用期间请保留此窗口。停止时按 Ctrl+C。
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
pause
