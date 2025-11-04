
PDF_NOTEBOOK = train_model/tech_challenge_tuberculose.ipynb
OUT_BASE = $(basename $(notdir $(PDF_NOTEBOOK)))
LATEX_FILE = $(OUT_BASE).tex
PDF_FILE = $(OUT_BASE).pdf

.PHONY: pdf
pdf: $(LATEX_FILE)
	@echo "Sanitizing LaTeX (remove fancyhdr/header lines)"
	python3 tools/generate_pdf.py $(LATEX_FILE)
	@echo "Building PDF (xelatex x3)"
	xelatex -interaction=nonstopmode $(LATEX_FILE) >/dev/null || true
	xelatex -interaction=nonstopmode $(LATEX_FILE) >/dev/null || true
	xelatex -interaction=nonstopmode $(LATEX_FILE) >/dev/null || true
	@echo "PDF build finished. Check $(PDF_FILE)"

$(LATEX_FILE): $(PDF_NOTEBOOK)
	@echo "Converting notebook to LaTeX"
	# ensure output filename matches OUT_BASE
	jupyter nbconvert --to latex --output "$(OUT_BASE)" --output-dir=. $(PDF_NOTEBOOK)
