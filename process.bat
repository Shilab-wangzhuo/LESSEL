REM @echo off
setlocal

REM 初始化输入文件和输出目录的变量
set "input_file="
set "output_dir="

REM 解析命令行参数
:parse_args
if "%~1"=="" goto check_input
if "%~1"=="-i" (
    set "input_file=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="-o" (
    set "output_dir=%~2"
    shift
    shift
    goto parse_args
)
echo Invalid option: %~1
exit /b 1

REM 检查输入文件参数
:check_input
if "%input_file%"=="" (
    echo Usage: %~nx0 -i ^<input_file^> [-o ^<output_dir^>]
    exit /b 1
)

REM 将相对路径转换为绝对路径
for %%I in ("%input_file%") do set "input_file=%%~fI"

REM 如果未提供输出目录，则默认为输入文件所在目录，并以输入文件名（不包含后缀）作为新目录名
if "%output_dir%"=="" (
    for %%I in ("%input_file%") do (
        set "output_dir=%%~dpnI"
    )
)

REM 获取文件名
for %%I in ("%input_file%") do set "name=%%~nI"

REM 创建输出目录
if not exist "%output_dir%" (
    mkdir "%output_dir%"
)

REM 创建 Step1 ~ Step7 文件夹
for %%S in (Step1_slide Step2_YOLOX Step3_sc_slide Step4_qc Step5_cut Step6_classify) do (
    mkdir "%output_dir%\%%S"
)

echo Folders created successfully in: %output_dir%

REM 定义日志文件路径
set log_file="%output_dir%\log_file.txt"


REM 激活虚拟环境
call conda activate learn >> %log_file% 2>&1

REM 运行 Python 脚本
python tools\Step1_slide\svs_slide.py -i "%input_file%" -o "%output_dir%\Step1_slide" >> %log_file% 2>&1

python tools\Step2_YOLOX\YOLOX\tools\demo1.py image -n yolox-x -c tools\Step2_YOLOX\YOLOX\YOLOX_outputs\yolox_voc_s\best_ckpt.pth --path "%output_dir%\Step1_slide\%name%" --save_dir "%output_dir%\Step2_YOLOX" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device gpu >> %log_file% 2>&1

python tools\Step3_sc_slide\sc_slide.py "%output_dir%\Step2_YOLOX\%name%"  "%output_dir%\Step1_slide\%name%"  "%output_dir%\Step3_sc_slide" >> %log_file% 2>&1

python tools\Step4_qc\QC.py --test_dir "%output_dir%\Step3_sc_slide\%name%"  --save_dir "%output_dir%\Step4_qc\%name%" >> %log_file% 2>&1

REM 激活虚拟环境
call conda activate myenv >> %log_file% 2>&1

python tools\Step5_cut\Pytorch-UNet-master\predict.py -i "%output_dir%\Step4_qc\%name%" -o "%output_dir%\Step5_cut" >> %log_file% 2>&1

REM 激活虚拟环境
call conda activate learn >> %log_file% 2>&1

python tools\Step6_classify\efficient_classify.py --test_dir "%output_dir%\Step5_cut\%name%\json_cut_out"   --save_dir "%output_dir%\Step6_classify\%name%" --ori_img_dir "%output_dir%\Step4_qc\%name%" >> %log_file% 2>&1


REM 结束
