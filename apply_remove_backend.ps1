# apply_remove_backend.ps1
$ErrorActionPreference = "Stop"

$workspace = "C:\ion_chronos\workspace"
if (-not (Test-Path $workspace)) {
  Write-Error "Workspace path not found: $workspace"
  exit 1
}

# 1) Backup entire workspace to a zip (fast recovery)
$ts = Get-Date -Format yyyyMMdd_HHmmss
$backupzip = "C:\ion_chronos\workspace_backup_$ts.zip"
Write-Output "Creating full workspace ZIP backup: $backupzip"
Compress-Archive -Path "$workspace\*" -DestinationPath $backupzip -Force

# 2) Create archive folder for moved backend(s)
$archive = Join-Path $workspace ("archive_removed_backend_$ts")
New-Item -ItemType Directory -Path $archive | Out-Null
Write-Output "Archive folder: $archive"

# 3) Move top-level backend if it exists
$top_backend = Join-Path $workspace "backend"
if (Test-Path $top_backend) {
  $dest = Join-Path $archive "backend_top"
  Move-Item -Path $top_backend -Destination $dest -Force
  Write-Output "Moved: $top_backend -> $dest"
} else {
  Write-Output "No top-level backend found at: $top_backend"
}

# 4) Move nested backend if it exists
$nested_backend = Join-Path $workspace "workspace\backend"
if (Test-Path $nested_backend) {
  $dest2 = Join-Path $archive "backend_nested"
  # ensure destination parent exists
  New-Item -ItemType Directory -Path $dest2 -Force | Out-Null
  Move-Item -Path $nested_backend -Destination $dest2 -Force
  Write-Output "Moved: $nested_backend -> $dest2"
} else {
  Write-Output "No nested backend found at: $nested_backend"
}

# 5) Optional: Report any docker-compose files
$composeTop = Join-Path $workspace "docker-compose.yml"
$composeNested = Join-Path $workspace "workspace\docker-compose.yml"
if (Test-Path $composeTop) { Write-Output "Found docker-compose at: $composeTop (not moved)" }
if (Test-Path $composeNested) { Write-Output "Found nested docker-compose at: $composeNested (not moved)" }

# 6) Final listing of archive contents
Write-Output "`nArchive contents:"
Get-ChildItem -Path $archive -Recurse | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize

Write-Output "`nBackup and move complete. To restore, copy files from the archive folder back to their original locations or extract the zip: $backupzip"
