param(
    [string]$Python = 'D:\proiect\rgbd-head-research-runtime-20261030\env-b\python.exe',
    [string]$Sequence = 'D:\proiect\rgbd-head-research-runtime-20261030\sequence-v2\sequence.json',
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda',
    [string]$B = 'D:\proiect\rgbd-head-reconstruction-runtime-20261030\replay-b-original',
    [string]$BFront = 'D:\proiect\rgbd-head-reconstruction-runtime-20261030\replay-b-front-original',
    [string]$OriginalComparison = 'D:\proiect\rgbd-head-reconstruction-runtime-20261030\comparison-replay-original\comparison.json'
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw 'Output must be a new isolated directory.' }
New-Item -ItemType Directory -Path $Output | Out-Null
$Entry = $PSScriptRoot
function Invoke-RecordedPython {
    param([string[]]$PythonArgs)
    Write-Host ('COMMAND: & "{0}" -B {1}' -f $Python, (($PythonArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '))
    & $Python -B @PythonArgs
    $Code = $LASTEXITCODE
    Write-Host "EXIT: $Code"
    if ($Code -ne 0) { throw "Python stage failed with exit $Code; outputs preserved." }
}
# Run sequentially on an otherwise idle GPU. This script never kills processes.
Invoke-RecordedPython -PythonArgs @('-m','unittest','discover','-s',$Entry,'-p','test_geometry*.py','-v')
$Front = Join-Path $Output 'front'
$Guarded = Join-Path $Output 'guarded'
Invoke-RecordedPython -PythonArgs @((Join-Path $Entry 'fit_c.py'),'--sequence',$Sequence,'--output',$Front,'--device',$Device,'--steps','240','--points','1200','--expression-components','30','--front-only')
Invoke-RecordedPython -PythonArgs @((Join-Path $Entry 'fit_c.py'),'--sequence',$Sequence,'--output',$Guarded,'--device',$Device,'--steps','240','--points','1200','--expression-components','30','--front-anchor',$Front)
Invoke-RecordedPython -PythonArgs @((Join-Path $Entry 'evaluate_c.py'),'--sequence',$Sequence,'--original-comparison',$OriginalComparison,'--b',$B,'--b-front',$BFront,'--c',$Guarded,'--c-front',$Front,'--output',(Join-Path $Output 'comparison'))
foreach ($Fit in @($Front,$Guarded)) {
    Invoke-RecordedPython -PythonArgs @((Join-Path $Entry 'verify_c.py'),'--fit',$Fit,'--sequence',$Sequence,'--output',(Join-Path $Fit 'verification.json'))
}
