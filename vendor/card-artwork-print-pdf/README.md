# Bundled print-PDF components

This directory contains the two runtime components required by the exporter:

- `scripts/build_print_pdf.py` writes a conservative single-raster PDF 1.4 file.
- `scripts/verify_artwork.py` verifies page boxes, direct RGB embedding, exact
  embedded pixels, and a nonblank independent render.

They are loaded relative to the installed Fate Tarot Skill, never from the
current project directory. An alternative root is accepted only when explicitly
provided through `--print-skill-root` or `CARD_ARTWORK_PRINT_PDF_SKILL`.

Pinned source SHA-256 values:

```text
build_print_pdf.py  3824654e9d8e3044c85ffb19796fb849ca99d176aea00dff233cd18494802a24
verify_artwork.py   1cfc575667b7422605e388c40d14c060ef7e40ef39cebc67474729fc69027982
```
