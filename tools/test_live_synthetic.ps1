param([string]$ExampleDir, [string]$ShimDir, [switch]$PromptForKey)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($PromptForKey) {
    $env:LLM_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Paste OpenRouter API key' -AsSecureString)).Password.Trim()
}
if (-not $env:LLM_API_KEY) { throw 'LLM_API_KEY is absent in this terminal; keep your key local.' }
if (-not $env:LLM_BASE_URL) { $env:LLM_BASE_URL = 'https://openrouter.ai/api/v1' }
if (-not $env:LLM_MODEL) { $env:LLM_MODEL = 'nvidia/nemotron-3.5-super-vl:free' }
if (([uri]$env:LLM_BASE_URL).Host -ne 'openrouter.ai' -or -not $env:LLM_MODEL.EndsWith(':free')) {
    throw 'This rehearsal is restricted to an explicitly configured OpenRouter :free model.'
}
$reference = Join-Path (Get-Location) 'example'
if (-not (Test-Path -LiteralPath (Join-Path $reference 'chats.jsonl'))) {
    $reference = Join-Path (Get-Location) '.private/official-day1/cold-start/example'
}
if (-not $ExampleDir) { $ExampleDir = $reference }
$ExampleDir = (Resolve-Path -LiteralPath $ExampleDir).Path
# Refuse arbitrary real-data folders, even when a stale terminal variable points there.
foreach ($name in @('chats.jsonl','messages.jsonl','seed.json')) {
    if ((Get-FileHash -LiteralPath (Join-Path $ExampleDir $name)).Hash -ne
        (Get-FileHash -LiteralPath (Join-Path $reference $name)).Hash) {
        throw 'Input does not match the preserved official synthetic fixture.'
    }
}
$python = (Get-Command python -ErrorAction Stop).Source
$bash = Join-Path $env:ProgramFiles 'Git/bin/bash.exe'
if (-not (Test-Path -LiteralPath $bash)) { throw 'Git Bash was not found under Program Files/Git.' }
if (-not $ShimDir -or -not (Test-Path -LiteralPath (Join-Path $ShimDir 'python3'))) {
    $ShimDir = Join-Path ([IO.Path]::GetTempPath()) ('xpert-python-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $ShimDir | Out-Null
    $executable = $python.Replace('\','/').Replace("'", "'\''")
    $shim = "#!/usr/bin/env bash`nexec '" + $executable + "' `"`$@`"`n"
    [IO.File]::WriteAllText((Join-Path $ShimDir 'python3'), $shim, (New-Object Text.UTF8Encoding $false))
}
$env:LLM_REQUEST_TIMEOUT_SECONDS = '90'
# Controlled experiment, not an assumed diagnosis. Other configured free models
# retain their defaults; this client's opt-in is verified only for this exact route.
$env:LLM_DISABLE_REASONING = if ($env:LLM_MODEL -eq 'nvidia/nemotron-3.5-super-vl:free') { '1' } else { '0' }
$env:XPERT_PROGRESS = '1'
Write-Host ('Starting fresh synthetic test with ' + $env:LLM_MODEL)
Write-Host 'Stage and completed-request progress follows. A slow request can remain quiet for up to 90 seconds.'
$folder = Join-Path (Get-Location) ('.private/live-synthetic/' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $folder | Out-Null
$inputDir = Join-Path $folder 'input'
New-Item -ItemType Directory -Path $inputDir | Out-Null
foreach ($name in @('chats.jsonl','messages.jsonl','seed.json')) {
    Copy-Item -LiteralPath (Join-Path $ExampleDir $name) -Destination (Join-Path $inputDir $name)
}
$output = Join-Path $folder 'output.json'
$launcher = Join-Path $folder 'invoke.sh'
[IO.File]::WriteAllText($launcher, "#!/usr/bin/env bash`nexec ./run.sh `"`$@`"`n", (New-Object Text.UTF8Encoding $false))
$validator = Join-Path $folder 'validate.py'
@'
import json,sys
sys.path.insert(0,sys.argv[3])
from inputs import load_index
from validation import validate
i=load_index(sys.argv[1])
with open(sys.argv[2],encoding='utf-8') as stream: validate(json.load(stream),i.chats,i.messages)
print('Strict validation passed')
'@ | Set-Content -LiteralPath $validator -Encoding UTF8
$record = [ordered]@{model=$env:LLM_MODEL; reasoning_disabled=$env:LLM_DISABLE_REASONING; fixture='official synthetic'; run_exit=$null; elapsed_seconds=$null; checker_exit=$null; strict_validation_exit=$null; output=$output}
$oldPath = $env:PATH
$timer = [Diagnostics.Stopwatch]::StartNew()
try {
    $env:PATH = (Resolve-Path -LiteralPath $ShimDir).Path + ';' + $oldPath
    # Exact two-argument interface; no answer directory is passed to inference.
    & $bash $launcher $inputDir $output
    $record.run_exit = $LASTEXITCODE
    $timer.Stop(); $record.elapsed_seconds = [math]::Round($timer.Elapsed.TotalSeconds,3)
    if ($record.run_exit -eq 0 -and (Test-Path -LiteralPath $output)) {
        & $python $validator $inputDir $output (Get-Location).Path
        $record.strict_validation_exit = $LASTEXITCODE
        # Only the checker receives labels, after inference has finished.
        & $python check.py $ExampleDir $output
        $record.checker_exit = $LASTEXITCODE
    }
} finally {
    $env:PATH = $oldPath
    $record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'run-summary.json') -Encoding UTF8
}
$record | ConvertTo-Json
if (Test-Path -LiteralPath ($output + '.run-status.json')) {
    Get-Content -LiteralPath ($output + '.run-status.json') -Raw | ConvertFrom-Json |
        Select-Object status,stage,error_type,source_file,source_line,code_frames |
        ConvertTo-Json -Depth 4
}
if (Test-Path -LiteralPath ($output + '.usage.jsonl')) {
    Get-Content -LiteralPath ($output + '.usage.jsonl') | ForEach-Object {
        $_ | ConvertFrom-Json | Select-Object request_id,stage,status,failure_kind,finish_reason,max_tokens,reasoning_tokens,prompt_tokens,completion_tokens,total_tokens,seconds,http_status,remaining_run_seconds
    } | ConvertTo-Json -Depth 3
}
if (Test-Path -LiteralPath ($output + '.completion.json')) {
    Get-Content -LiteralPath ($output + '.completion.json') -Raw | ConvertFrom-Json |
        Select-Object status,reasons,coverage | ConvertTo-Json -Depth 8
}
if ($record.run_exit -ne 0 -or $record.checker_exit -ne 0 -or $record.strict_validation_exit -ne 0) {
    throw 'Live synthetic test did not pass. Safe diagnostics above; private artifacts retained.'
}
Write-Host 'Fresh live synthetic output passed strict validation and the project checker.'
