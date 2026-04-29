TEX = main

.PHONY: all clean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex preprint.cls references.bib
	pdflatex $(TEX)
	-bibtex $(TEX)
	pdflatex $(TEX)
	pdflatex $(TEX)

clean:
	rm -f $(TEX).{pdf,aux,bbl,blg,log,out,toc,fls,fdb_latexmk,synctex.gz}
