@echo off
pushd "%~dp0\..\..\.."
"D:\app\anna\envs\cherry\python.exe" "02_code\05_utils\pruning_truth\prepare_pruning_segment_cleaning.py" --manifest "01_data\03_processed\pruning_gnn_v1\manifests\current_pipeline_manifest.json" --output "04_results\pruning_decision\pruning_segment_cleaning_v2_reference_budflow" --device cuda:0 --resume --retry-failed
pause
popd
