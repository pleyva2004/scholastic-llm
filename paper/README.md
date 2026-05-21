# scholastic-llm paper

This directory contains the LaTeX source for the project preprint:
*Teaching a Small LLM Scholastic Voice: Fine-Tuning Qwen 2.5 on the
Catechism, Summa, and Augustine via Local MLX.*

## Building locally

You need a working TeX Live (or MacTeX) installation with `latexmk`,
`pdflatex`, `pgfplots`, `tikz`, `booktabs`, `hyperref`, `titlesec`, and
`microtype`. These are all standard in a full TeX Live install.

From the repository root:

```bash
cd paper
latexmk -pdf main.tex
```

`latexmk` will run `pdflatex` and `bibtex` enough times to resolve all
cross-references. The output PDF lives at `paper/main.pdf`.

To clean up intermediate files:

```bash
latexmk -C
```

## Building on GitHub Actions

Every push to `main` that touches anything under `paper/` triggers the
`Build paper` workflow defined in `.github/workflows/build-paper.yml`. The
workflow:

1. Checks out the repository.
2. Compiles `paper/main.tex` with the `xu-cheng/latex-action` container
   (full TeX Live + latexmk).
3. Uploads the resulting `paper/main.pdf` as an artifact named
   `scholastic-llm-paper`.
4. If the push is a tag, attaches the PDF to the GitHub release for that
   tag.

You can also trigger the workflow manually from the GitHub Actions UI
(`workflow_dispatch`).

## File layout

```
paper/
  main.tex                 # top-level document
  abstract.tex             # abstract block
  arxiv-style.sty          # local arxiv-style preprint package
  references.bib           # BibTeX bibliography
  sections/
    01-introduction.tex
    02-related-work.tex
    03-method.tex
    04-experiments.tex
    05-discussion.tex
    06-conclusion.tex
  figures/
    pipeline.tex           # TikZ data pipeline diagram
    training-curve.tex     # pgfplots training/val loss curve
    rubric-bars.tex        # pgfplots grouped rubric bar chart
  README.md                # this file
```
