# Poster

A0 portrait (84.1 × 118.9 cm) conference poster built with the
[Gemini](https://github.com/anishathalye/gemini) beamerposter theme
(MIT, vendored into this directory).

## Local build

```bash
cd poster
latexmk -pdf main.tex
# → poster/main.pdf
```

Requires a TeX Live distribution with `beamer`, `beamerposter`, `pgfplots`,
`tikz`, `qrcode`. All are standard in TeX Live's `texlive-full`.

## Layout

- 2 columns
- Left: Motivation · Pipeline · Training setup
- Right: Results (rubric bars) · Qualitative example · Takeaway · Phase 2
  status · Resources + QR

## CI

Pushes to `poster/**` on `main` trigger
[`.github/workflows/build-poster.yml`](../.github/workflows/build-poster.yml),
which compiles the poster and commits the rendered PDF to `docs/poster.pdf`,
making it live at
[https://pleyva2004.github.io/scholastic-llm/poster.pdf](https://pleyva2004.github.io/scholastic-llm/poster.pdf).
