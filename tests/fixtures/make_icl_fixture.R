# Reproducibility documentation, not a CI dependency (R/mclust is not
# installed in the tgmm dev environment this was written against -- see
# docs/superpowers/specs/2026-10-14-phase1-metrics-model-selection-design.md).
#
# Generates tests/fixtures/mclust_icl_fixture.json: a seeded 3-blob 2-D
# dataset, mclust's fitted VVV (full-covariance) parameters at G=3, and the
# icl() value mclust itself reports for that fit. tests/test_phase1.py's
# test_icl_matches_mclust_fixture loads this JSON, sets a tgmm
# GaussianMixture's weights_/means_/covariances_ directly to mclust's
# fitted values (no re-fitting), and checks tgmm's icl() against mclust's.
#
# Run with: Rscript tests/fixtures/make_icl_fixture.R
# Requires: install.packages(c("mclust", "jsonlite"))

library(mclust)
library(jsonlite)

set.seed(0)
n_per <- 80
centers <- rbind(c(0, 0), c(6, 0), c(0, 6))
X <- do.call(rbind, lapply(1:nrow(centers), function(i) {
  sweep(matrix(rnorm(n_per * 2), ncol = 2), 2, centers[i, ], `+`)
}))

fit <- Mclust(X, G = 3, modelNames = "VVV")

# mclust's icl() function (mclust >= 5.4) returns the ICL table; pull out
# the VVV/G=3 entry to match `fit` exactly.
icl_table <- icl(Mclust(X, G = 3, modelNames = "VVV", x = mclustBIC(X, G = 3, modelNames = "VVV")))
icl_value <- as.numeric(icl_table[1, 1])

fixture <- list(
  X = X,
  weights = fit$parameters$pro,
  means = t(fit$parameters$mean),             # mclust stores means as (d, K); tgmm wants (K, d)
  covariances = aperm(fit$parameters$variance$sigma, c(3, 1, 2)),  # (d, d, K) -> (K, d, d)
  icl = icl_value
)

write_json(fixture, "tests/fixtures/mclust_icl_fixture.json", digits = 16, auto_unbox = TRUE)
