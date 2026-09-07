@echo off
pushd "%~dp0\..\..\.."
"D:\app\anna\envs\cherry\python.exe" "02_code\05_utils\pruning_truth\export_cleaned_pruning_gnn_dataset.py" --dataset "04_results\pruning_decision\pruning_segment_cleaning_v1"
pause
popd
