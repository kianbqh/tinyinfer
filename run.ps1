param(
    [ValidateSet('cpu', 'gpu')][string]$Mode = 'cpu',
    [string]$Distribution,
    [string]$CudaHome
)

$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot.Replace('\', '/')
if ($projectPath -notmatch '^([A-Za-z]):/(.*)$') { throw 'Run the project from a drive mounted in WSL.' }
$wslPath = '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2]
$wslArgs = @()
if ($Distribution) { $wslArgs += @('-d', $Distribution) }
$wslArgs += @('--cd', $wslPath, '--', 'env')
if ($CudaHome) { $wslArgs += "CUDACXX=$CudaHome/bin/nvcc" }
$wslArgs += @('bash', "scripts/run_$Mode.sh")
& wsl @wslArgs
if ($LASTEXITCODE -ne 0) { throw "$Mode run exited with code $LASTEXITCODE" }
