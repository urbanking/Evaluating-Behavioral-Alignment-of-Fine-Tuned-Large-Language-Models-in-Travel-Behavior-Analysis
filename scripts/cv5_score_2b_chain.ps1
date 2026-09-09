# Score qwen_mid (2B) CV5 folds in order.
# ASCII-only on purpose: a BOM-less UTF-8 .ps1 is read as CP949 by PowerShell 5.1
# on this machine and Korean comments broke the parser (2026-09-02).
#
# Exit-code trap (2026-09-02): Start-Process -PassThru returns a Process whose
# ExitCode is $null unless the handle is cached first. $null -ne 0 is TRUE in
# PowerShell, so the old guard stopped the chain after a SUCCESSFUL fold 1.
# Fix: cache $p.Handle before WaitForExit, and treat the written parquet as the
# real success criterion.
#
# Folds whose parquet already exists are skipped, so this is resumable.
$ErrorActionPreference='Continue'
$ROOT='C:\SP_LLM\TRC'; $PY='C:\gpu-work\.venv\Scripts\python.exe'
Set-Location $ROOT
$LOG = Join-Path $ROOT 'logs\cv5_score_2b.out'
function Say($m) { ('[' + (Get-Date -Format 'MM-dd HH:mm:ss') + '] ' + $m) | Out-File -Append -Encoding utf8 $LOG }
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
$env:TRC_MODEL_ROOT='C:/SP_LLM/models'
$env:TRC_DTYPE='float16'; $env:CUDA_VISIBLE_DEVICES='0'
$env:TRC_MEM_FRACTION='0.9'
$env:PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.8'
# 2026-09-02: fold 4 died with "forrtl: error (200): program aborting due to
# window-CLOSE event". That is the Intel Fortran runtime (via numpy/scipy MKL)
# aborting on a console CTRL_CLOSE event. This switch stops it from doing that.
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER='1'
Say '======== 2B CV5 scoring start ========'
foreach ($f in 1,2,3,4,5) {
  $par = Join-Path $ROOT ('predictions\cv5_llm\qwen_mid_f' + $f + '_base.parquet')
  if (Test-Path $par) { Say ('[skip] fold ' + $f + ' already scored'); continue }
  Say ('[run ] fold ' + $f)
  $t0 = Get-Date
  $p = Start-Process -FilePath $PY -NoNewWindow -PassThru -WorkingDirectory $ROOT -RedirectStandardOutput (Join-Path $ROOT ('logs\cv5_score_2b_f' + $f + '.out')) -RedirectStandardError (Join-Path $ROOT ('logs\cv5_score_2b_f' + $f + '.err')) -ArgumentList @('-u','scripts\cv5_score_fold.py','--family','qwen_mid','--fold',"$f")
  $null = $p.Handle
  $p.WaitForExit()
  $code = $p.ExitCode
  $min = [math]::Round(((Get-Date)-$t0).TotalMinutes,1)
  $wrote = Test-Path $par
  Say ('[end ] fold ' + $f + '  exit=' + $(if ($null -eq $code) { 'null' } else { $code }) + '  parquet=' + $wrote + '  ' + $min + ' min')
  if (-not $wrote) { Say ('[stop] fold ' + $f + ' wrote no parquet. stopping the chain.'); break }
}
Say '======== 2B CV5 scoring done ========'