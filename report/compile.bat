# compile.ps1 — Run this in PowerShell from anywhere
# Usage: cd report; .\compile.ps1  OR  just double-click compile.bat

$ErrorActionPreference = "Continue"
$ReportDir = $PSScriptRoot
if (-not $ReportDir) { $ReportDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $ReportDir) { $ReportDir = Get-Location }

Set-Location $ReportDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Compiling LaTeX Report" -ForegroundColor Cyan
Write-Host "  Directory: $ReportDir" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "`n[1/4] pdflatex (first pass) ..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode main.tex | Select-String "Error|Warning|!" | Write-Host

Write-Host "`n[2/4] bibtex (references) ..." -ForegroundColor Yellow
bibtex main

Write-Host "`n[3/4] pdflatex (second pass) ..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode main.tex | Out-Null

Write-Host "`n[4/4] pdflatex (final pass) ..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode main.tex | Out-Null

if (Test-Path "main.pdf") {
    $size = [math]::Round((Get-Item "main.pdf").Length / 1KB, 1)
    Write-Host "`nDone. Output: $ReportDir\main.pdf  ($size KB)" -ForegroundColor Green
    Start-Process "main.pdf"
} else {
    Write-Host "`nERROR: main.pdf not generated. Check main.log for details." -ForegroundColor Red
}
