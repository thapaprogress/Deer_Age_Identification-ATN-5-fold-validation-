Write-Host "============================================="
Write-Host "   ATN Auto-Scheduler"
Write-Host "============================================="
Write-Host "Monitoring training process..."

# Loop until training is done
while ($true) {
    # Check for python process running train.py
    # Using WMI to get command line arguments
    $process = Get-WmiObject Win32_Process -Filter "name = 'python.exe' AND commandline LIKE '%train.py%'"
    
    if (-not $process) {
        Write-Host "Training process has finished (or was not found)."
        break
    }
    
    $current_time = Get-Date -Format "HH:mm:ss"
    Write-Host "[$current_time] Training still in progress... Checking again in 60 seconds."
    Start-Sleep -Seconds 60
}

# Training done, start evaluation
Write-Host "`n============================================="
Write-Host "Training Finished! Starting Evaluation..."
Write-Host "============================================="

# Run evaluation
python training/evaluate.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "Evaluation completed successfully."
} else {
    Write-Host "Error during evaluation."
}

# Run ONNX export
Write-Host "`n============================================="
Write-Host "Starting ONNX Model Export..."
Write-Host "============================================="
python utils/export_onnx.py

Write-Host "`n============================================="
Write-Host "   ALL TASKS COMPLETED"
Write-Host "============================================="
Read-Host -Prompt "Press Enter to exit"
