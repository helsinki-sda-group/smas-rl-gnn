# slurm_submit.ps1 - PowerShell wrapper for Slurm submission
# Usage on local machine for testing/building commands:
#   .\slurm_submit.ps1 1hop_rnd 2hop 1hop_ctc -DryRun
#
# To execute on Mahti:
#   ssh mahti "cd /projappl/project_2012159/kbocheni_temp/smas-rl-gnn && bash slurm/slurm_submit.sh 1hop_rnd 2hop 1hop_ctc"

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

# Parse arguments into methods and options
$methods = @()
$mode = "new"
$dryRun = $false

$i = 0
while ($i -lt $Arguments.Count) {
    $arg = $Arguments[$i]
    switch ($arg) {
        "--mode" {
            if ($i + 1 -lt $Arguments.Count) {
                $mode = $Arguments[$i + 1]
                $i += 2
            } else {
                $i++
            }
        }
        "--dry-run" {
            $dryRun = $true
            $i++
        }
        default {
            $methods += $arg
            $i++
        }
    }
}

if ($methods.Count -eq 0) {
    Write-Host "[ERROR] No methods specified. Usage: .\slurm_submit.ps1 METHOD1 [METHOD2 ...] [--mode new|continue] [--dry-run]" -ForegroundColor Red
    exit 1
}

# Get the parent directory (project root)
$slurm_dir = Split-Path -Parent $PSCommandPath
$project_root = Split-Path -Parent $slurm_dir
$config_dir = Join-Path $project_root "configs"

if (-not (Test-Path $config_dir)) {
    Write-Host "[ERROR] Config directory not found: $config_dir" -ForegroundColor Red
    exit 1
}

# Find all YAML config files
$allYamls = @(Get-ChildItem -Path $config_dir -Filter "rp_gnn_*.yaml" -File | Sort-Object Name)

function Match-MethodToConfigs {
    param([string]$Method)
    
    $matched = @()
    
    # Parse method pattern
    if ($Method -match '^(.+)_([a-z]+)$') {
        # Has variant (e.g., "1hop_rnd" -> base="1hop", variant="rnd")
        $base = $Matches[1]
        $variant = $Matches[2]
        $pattern = "rp_gnn_${base}-[0-9]*_${variant}.yaml"
        
        foreach ($yaml in $allYamls) {
            if ($yaml.Name -like $pattern) {
                $matched += $yaml
            }
        }
    } else {
        # No variant (e.g., "2hop" -> match only base configs without variant suffix)
        $base = $Method
        $pattern = "rp_gnn_${base}-[0-9]*.yaml"
        
        foreach ($yaml in $allYamls) {
            # Include only if it matches pattern AND doesn't have underscore (variant suffix)
            if ($yaml.Name -like $pattern -and $yaml.Name -notmatch '_[a-z]+\.yaml$') {
                $matched += $yaml
            }
        }
    }
    
    return $matched
}

Write-Host "[INFO] Slurm Job Submission Script" -ForegroundColor Cyan
Write-Host "  Mode: $mode"
$dryRunStr = if ($dryRun) { 'enabled' } else { 'disabled' }
Write-Host "  Dry-run: $dryRunStr"
Write-Host ""

$totalJobs = 0
$submittedCount = 0

foreach ($method in $methods) {
    $configs = Match-MethodToConfigs -Method $method
    
    if ($configs.Count -eq 0) {
        Write-Host "[WARN] No configs found for method: $method" -ForegroundColor Yellow
        continue
    }
    
    Write-Host "[INFO] Found $($configs.Count) config(s) for method: $method" -ForegroundColor Green
    
    foreach ($config in $configs) {
        $totalJobs++
        $configRel = "configs/$($config.Name)"
        
        # Extract run_name from config
        $runName = (Select-String -Path $config.FullName -Pattern 'run_name:\s*(.+)' | ForEach-Object { $_.Matches[0].Groups[1].Value }).Trim()
        if (!$runName) { $runName = "unknown" }
        
        if ($dryRun) {
            Write-Host "  [DRY-RUN] sbatch --job-name='rp-train-$runName' run_train.sbatch '$configRel' '$mode'" -ForegroundColor Cyan
        } else {
            Write-Host "  [SUBMIT] $($config.Name) (run_name=$runName)" -ForegroundColor Cyan
            Write-Host "    Command: ssh mahti 'cd /projappl/project_2012159/kbocheni_temp/smas-rl-gnn && sbatch --job-name=rp-train-$runName slurm/run_train.sbatch $configRel $mode'" -ForegroundColor Gray
        }
        $submittedCount++
    }
}

Write-Host ""
Write-Host "[INFO] Submission Summary" -ForegroundColor Cyan
Write-Host "  Total configs: $totalJobs"
Write-Host "  Ready to submit: $submittedCount"
Write-Host ""

if (-not $dryRun) {
    Write-Host "[HINT] Copy the command above and paste into Mahti terminal to submit" -ForegroundColor Yellow
}
