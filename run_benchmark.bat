@echo off
REM Benchmark script for DEIM models

REM Check for Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.8 or later.
    exit /b 1
)

REM Parse command line arguments
set MODEL_PATH=%1
set INPUT_PATH=%2

if "%MODEL_PATH%"=="" (
    echo Usage: run_benchmark.bat MODEL_PATH INPUT_PATH [options]
    echo.
    echo Required arguments:
    echo   MODEL_PATH  - Path to the model file
    echo   INPUT_PATH  - Path to input image directory or video file
    echo.
    echo Options:
    echo   --input-type [image^|video] - Type of input (default: image)
    echo   --model-type [deim^|yolo]   - Type of model (default: deim)
    echo   --optimize                 - Optimize model before benchmarking
    echo   --no-visualization         - Disable saving visualizations
    echo   --device [cpu^|cuda^|auto]   - Device to run on (default: auto)
    exit /b 1
)

if "%INPUT_PATH%"=="" (
    echo Error: INPUT_PATH is required
    exit /b 1
)

echo Running benchmark with model: %MODEL_PATH%
echo Input: %INPUT_PATH%

REM Execute the benchmark script with all arguments
python benchmark_image.py --model-path %MODEL_PATH% --input %INPUT_PATH% %3 %4 %5 %6 %7 %8 %9

echo.
echo Benchmark completed! 