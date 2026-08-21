@echo off
rem 易碎类四阶段顺序采集: 轻脆 -> 重脆 -> E系易碎补充 -> FRA柔性
set PYTHONPATH=E:\Lib\site-packages
set LOGDIR=E:\A-触觉机器学习\数据采集准备
E:\python.exe "%LOGDIR%\collect_wikimedia.py" --phase fra_light  >> "%LOGDIR%\collect_fra.log" 2>&1
E:\python.exe "%LOGDIR%\collect_wikimedia.py" --phase fra_heavy  >> "%LOGDIR%\collect_fra.log" 2>&1
E:\python.exe "%LOGDIR%\collect_wikimedia.py" --phase e_fragile  >> "%LOGDIR%\collect_fra.log" 2>&1
E:\python.exe "%LOGDIR%\collect_wikimedia.py" --phase fra_soft   >> "%LOGDIR%\collect_fra.log" 2>&1
echo ALL_FRAGILE_DONE >> "%LOGDIR%\collect_fra.log"
