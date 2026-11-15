#!/usr/bin/env Rscript
# E02 · FateID and STEMNET on the exported dataset (R2.1, R2.2).
#
# Usage:
#   Rscript revision/E02_baselines/fateid_stemnet.R revision/results/E02/<dataset>
#
# Expects <dir>/r_input/ from export_for_r.py. Writes <dir>/fateid.csv and
# <dir>/stemnet.csv in the shared prediction format (cell index, prob_<class>,
# pred). Both methods train on terminal cells and predict the query cells,
# exactly the setting Ageas is evaluated in.
#
# Packages: FateID (CRAN), Matrix; STEMNET (github: lsteinmetz/STEMNET).
# Check argument names against the installed versions before the final run.

suppressPackageStartupMessages({
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) > 0) args[1] else stop("pass the E02 results directory")
input <- file.path(dir, "r_input")

x <- readMM(file.path(input, "matrix.mtx"))
rownames(x) <- readLines(file.path(input, "features.txt"))
colnames(x) <- readLines(file.path(input, "cells.txt"))
obs <- read.csv(file.path(input, "obs.csv"), row.names = 1)
terminal <- setdiff(unique(obs$state), "query")
query_cells <- rownames(obs)[obs$is_query == "True" | obs$is_query == TRUE]

write_preds <- function(prob, path) {
  prob <- prob[query_cells, terminal, drop = FALSE]
  prob <- prob / pmax(rowSums(prob), 1e-12)
  out <- data.frame(prob, check.names = FALSE)
  colnames(out) <- paste0("prob_", terminal)
  out$pred <- terminal[max.col(prob, ties.method = "first")]
  write.csv(out, path)
}

# ---- FateID ---------------------------------------------------------------
# fateBias() needs a cluster partition y (integer per cell) and target
# clusters tar. Terminal classes become clusters 1..k, query cells k+1.
if (requireNamespace("FateID", quietly = TRUE)) {
  y <- setNames(match(obs$state, c(terminal, "query")), rownames(obs))
  tar <- seq_along(terminal)
  xd <- as.data.frame(as.matrix(x))
  fb <- FateID::fateBias(xd, y, tar, z = NULL, minnr = 5, minnrh = 10,
                         adapt = TRUE, confidence = 0.75, nbfactor = 5,
                         use.dist = FALSE, seed = 12345, nbtree = NULL)
  prob <- as.matrix(fb$probs)
  colnames(prob) <- terminal[as.integer(sub("^t", "", colnames(prob)))]
  write_preds(prob, file.path(dir, "fateid.csv"))
  message("FateID done")
} else {
  message("FateID not installed; skipped")
}

# ---- STEMNET --------------------------------------------------------------
# runSTEMNET() takes cells x genes and a population vector with NA for the
# cells to classify (the query progenitors).
if (requireNamespace("STEMNET", quietly = TRUE)) {
  population <- setNames(ifelse(obs$state == "query", NA, obs$state), rownames(obs))
  res <- STEMNET::runSTEMNET(t(as.matrix(x)), population)
  prob <- as.matrix(res@posteriors)
  write_preds(prob, file.path(dir, "stemnet.csv"))
  message("STEMNET done")
} else {
  message("STEMNET not installed; skipped")
}
