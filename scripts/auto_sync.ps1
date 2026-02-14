# Background Auto-Sync Script
Write-Host "Starting Auto-Sync Service..."

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] Checking for changes..."
    
    # Check if there are changes
    $status = git status --porcelain
    if ($status) {
        Write-Host "Changes detected. Syncing..."
        git add .
        git commit -m "Auto-sync: $timestamp"
        git push
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Sync successful."
        }
        else {
            Write-Host "Sync failed (check remote/network)."
        }
    }
    else {
        Write-Host "No changes detected."
    }
    
    # Wait for 1 hour (3600 seconds)
    Start-Sleep -Seconds 3600
}
