TEX = main
NIX = nix develop --command

.PHONY: all clean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex preprint.cls references.bib
	$(NIX) pdflatex $(TEX)
	-$(NIX) bibtex $(TEX)
	$(NIX) pdflatex $(TEX)
	$(NIX) pdflatex $(TEX)

clean:
	rm -f $(TEX).{pdf,aux,bbl,blg,log,out,toc,fls,fdb_latexmk,synctex.gz}
