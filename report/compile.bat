@echo off
echo ============================================================
echo  Compiling LaTeX Report
echo ============================================================
cd /d %~dp0
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo.
echo Done. Output: main.pdf
pause
