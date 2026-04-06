@echo off
chcp 65001 >nul
echo ============================================
echo 干滩识别系统 - 虚拟环境启动器
echo ============================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [错误] 虚拟环境未找到！
    echo 请先运行: python -m venv venv
    echo 然后安装依赖: venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [1] 启动图形界面 (GUI)
echo [2] 命令行模式
echo [3] 生成测试数据
echo [4] 运行测试
echo [5] 退出
echo.

set /p choice=请选择 (1-5):

if "%choice%"=="1" goto gui
if "%choice%"=="2" goto cmd
if "%choice%"=="3" goto gen_data
if "%choice%"=="4" goto test
if "%choice%"=="5" goto end

:gui
echo.
echo 启动GUI界面...
venv\Scripts\python main.py gui
goto end

:cmd
echo.
echo 进入命令行模式 (输入 exit 退出)...
venv\Scripts\python -i
goto end

:gen_data
echo.
echo 生成测试数据...
venv\Scripts\python utils\generate_data.py --num 10 --output data\test
goto end

:test
echo.
echo 运行测试...
venv\Scripts\python test_project.py
goto end

:end
pause
