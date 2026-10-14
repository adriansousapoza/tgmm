# Reproducibility documentation, not a CI dependency (R/mclust is not
# installed in the tgmm dev environment this was written against -- see
# docs/superpowers/specs/2026-10-14-phase3-gpcm-covariance-models-design.md
# and the session's [[v0_4_mclust_oracle_gap]] memory note).
#
# Generates tests/fixtures/mclust_gpcm_fixtures.json: three seeded datasets
# (d in {2, 4, 6}, K=3), a fixed k-means partition z, and for each of the
# 14 GPCM model names, the `sigma` array mclust::mstep(modelName, data, z)
# returns in $parameters$variance$sigma (converted from mclust's (d, d, K)
# layout to tgmm's (K, d, d)), plus nVarParams(modelName, data=list(...))
# for the parameter-count cross-check.
#
# tests/test_phase3.py's test_matches_mclust_fixture loads this JSON,
# rebuilds the same z/W/nk tgmm-side, calls mstep_gpcm, and compares
# cov_full to mclust's sigma (atol=1e-8 for closed-form models, 1e-5 for
# the 5 iterative ones, matching mclust's own inner tolerance -- run with
# itmax raised to 200 in that comparison per the plan).
#
# Run with: Rscript tests/fixtures/make_gpcm_fixtures.R
# Requires: install.packages(c("mclust", "jsonlite"))

library(mclust)
library(jsonlite)

MODEL_NAMES <- c(
  "EII", "VII", "EEI", "VEI", "EVI", "VVI",
  "EEE", "VEE", "EVE", "EEV", "VVE", "VEV", "EVV", "VVV"
)

make_dataset <- function(d, seed) {
  set.seed(seed)
  n_per <- 60
  centers <- matrix(0, nrow = 3, ncol = d)
  for (i in 1:3) centers[i, 1] <- (i - 1) * 4
  X <- do.call(rbind, lapply(1:3, function(i) {
    sweep(matrix(rnorm(n_per * d), ncol = d), 2, centers[i, ], `+`)
  }))
  X
}

make_entry_for_dataset <- function(X) {
  n <- nrow(X)
  km <- kmeans(X, centers = 3, nstart = 10)
  z <- km$cluster - 1L # 0-indexed, to match the Python side

  # mclust wants a hard z as an indicator matrix for mstep().
  z_matrix <- matrix(0, nrow = n, ncol = 3)
  for (i in 1:n) z_matrix[i, z[i] + 1] <- 1

  models <- list()
  for (name in MODEL_NAMES) {
    fit <- mstep(modelName = name, data = X, z = z_matrix,
                 control = emControl(itmax = 200, tol = 1e-10))
    sigma <- fit$parameters$variance$sigma # (d, d, K)
    sigma_kdd <- aperm(sigma, c(3, 1, 2)) # -> (K, d, d)
    n_params <- nVarParams(name, d = ncol(X), G = 3)
    models[[name]] <- list(sigma = sigma_kdd, n_params = n_params)
  }

  list(X = X, z = z, models = models)
}

fixtures <- list(
  d2 = make_entry_for_dataset(make_dataset(2, 0)),
  d4 = make_entry_for_dataset(make_dataset(4, 1)),
  d6 = make_entry_for_dataset(make_dataset(6, 2))
)

# Flatten "models" into the per-code "sigma"/"n_params" keys
# test_phase3.py expects directly under fixtures[[dataset]][[code]].
flat <- list()
for (key in names(fixtures)) {
  flat[[key]] <- list(X = fixtures[[key]]$X, z = fixtures[[key]]$z)
  for (name in MODEL_NAMES) {
    flat[[key]][[name]] <- fixtures[[key]]$models[[name]]
  }
}

write_json(flat, "tests/fixtures/mclust_gpcm_fixtures.json", digits = 16, auto_unbox = TRUE)
