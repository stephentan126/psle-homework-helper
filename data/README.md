# Data folder

This folder is empty in the repository on purpose. Everything the pipeline reads or writes here is either
copyrighted exam material, data derived from it, or private data from students, so it is excluded by
`.gitignore` and never published.

The exam papers were bought for this project and used for computational data analysis only, as the
Copyright Act 2021 (ss. 243-244) allows. Their text, images and answers are not redistributed.

## Layout the pipeline expects

| Path | Contents | Source |
|---|---|---|
| `School papers/<year>/*.pdf` | School practice papers (SA1, SA2, WA and similar) | Purchased, not included |
| `Yearly/*.pdf` | Past PSLE papers | Purchased, not included |
| `Hold out/` | Papers set aside as the frozen held-out set for the grounding experiment | Purchased, not included |
| `extracted/` | Rendered page images, both OCR readers' output, and intermediate files | Written by the extraction pipeline |
| `student_submissions/` | Photos uploaded through the app, raw and preprocessed | Written by the running app |

The question bank itself is stored in `psle.db` at the repository root, which is also excluded.

## Running without the papers

The app can be run without any of this data on 12 original PSLE-style questions written for this project.
Build that database with `python scripts/dev/seed_demo_db.py` and follow Option 2 in the main
[README](../README.md).
