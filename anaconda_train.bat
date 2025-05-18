@echo off
call conda activate yolox
cd /d "C:\Users\AD\Desktop\DETEKCE work\nová verze YOLOX\YOLOX-main"
set PYTHONPATH=%CD%
python tools/train.py -f exps/example/custom/yolox_s.py -d 1 -b 16 --fp16 -o
pause
