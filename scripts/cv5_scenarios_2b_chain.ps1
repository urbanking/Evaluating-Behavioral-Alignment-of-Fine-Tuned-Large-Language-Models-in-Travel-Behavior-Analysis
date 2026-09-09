# Score qwen_mid (2B) CV5 fold adapters over the 33-condition grid, folds 1..5 in order.
# ASCII-only + UTF-8 BOM on purpose (PowerShell 5.1 reads a BOM-less file as CP949 here).
#
# Resumable at two levels: the python script skips existing shards, and this chain skips
# a fold whose merged parquet already exists. One retry per fold, then stop the chain.
# Exit code: cache $p.Handle before WaitForExit or ExitCode is $null ($null -ne 0 is TRUE).
# FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 stops the Intel Fortran runtime (numpy MKL) from
# aborting on a console CTRL_CLOSE event (this killed fold 4 of the BASE run on 09-02).
$ErrorActionPreference='Continue'
$ROOT='C:\SP_LLM\TRC'; $PY='C:\gpu-work\.venv\Scripts\python.exe'
Set-Location $ROOT
$LOG = Join-Path $ROOT 'logs\cv5_scen_2b.out'
function Say($m) { ('[' + (Get-Date -Format 'MM-dd HH:mm:ss') + '] ' + $m) | Out-File -Append -Encoding utf8 $LOG }
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
$env:TRC_MODEL_ROOT='C:/SP_LLM/models'
$env:TRC_DTYPE='float16'; $env:CUDA_VISIBLE_DEVICES='0'
$env:TRC_MEM_FRACTION='0.9'
$env:PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.8'
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER='1'
Say '======== 2B CV5 33-condition scoring start ========'
foreach ($f in 1,2,3,4,5) {
  $merged = Join-Path $ROOT ('predictions\cv5_llm\qwen_mid_f' + $f + '_scenarios.parquet')
  if (Test-Path $merged) { Say ('[skip] fold ' + $f + ' merged parquet exists'); continue }
  $ok = $false
  foreach ($attempt in 1,2) {
    Say ('[run ] fold ' + $f + ' attempt ' + $attempt)
    $t0 = Get-Date
    $p = Start-Process -FilePath $PY -NoNewWindow -PassThru -WorkingDirectory $ROOT -RedirectStandardOutput (Join-Path $ROOT ('logs\cv5_scen_2b_f' + $f + '.out')) -RedirectStandardError (Join-Path $ROOT ('logs\cv5_scen_2b_f' + $f + '.err')) -ArgumentList @('-u','scripts\cv5_score_fold_scenarios.py','--family','qwen_mid','--fold',"$f")
    $null = $p.Handle
    $p.WaitForExit()
    $code = $p.ExitCode
    $min = [math]::Round(((Get-Date)-$t0).TotalMinutes,1)
    $wrote = Test-Path $merged
    Say ('[end ] fold ' + $f + ' attempt ' + $attempt + '  exit=' + $(if ($null -eq $code) {'null'} else {$code}) + '  merged=' + $wrote + '  ' + $min + ' min')
    if ($wrote) { $ok = $true; break }
    Say ('[retry] fold ' + $f + ' produced no merged parquet; shards are kept, retrying once')
    Start-Sleep -Seconds 30
  }
  if (-not $ok) { Say ('[stop] fold ' + $f + ' failed twice. stopping the chain.'); break }
}
Say '======== 2B CV5 33-condition scoring done ========'