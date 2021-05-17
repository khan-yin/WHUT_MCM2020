xelatex.exe -synctex=1 -interaction=nonstopmode "thesis".tex
bibtex.exe "thesis"
xelatex.exe -synctex=1 -interaction=nonstopmode "thesis".tex
