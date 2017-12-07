#!/usr/bin/env Rscript

devtools::install_github("statswithr/statsr@BayesFactor")
bookdown::render_book("index.Rmd", "bookdown::gitbook")
bookdown::render_book("index.Rmd", "bookdown::pdf_book")
