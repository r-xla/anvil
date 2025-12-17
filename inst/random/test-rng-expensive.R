# Expensive RNG tests for statistical verification
# They are slow, so we don't run them in the normal CI suite
# These tests are excluded from the main test suite for performance reasons.
# Run manually with: source(system.file("random", "test-rng-expensive.R", package = "anvil"))

library(anvil)

test_rnorm_statistical <- function() {
  cat("Testing rnorm statistical properties...\n")

  # Test normality with large sample
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f64", shape_out = c(200L, 300L, 400L))
  }
  g <- jit(f)
  out <- g()
  Z <- as_array(out[[2]])

  # Check no non-finite values
  stopifnot(all(is.finite(Z)))

  # Shapiro-Wilk test on slices

  tst <- apply(Z, c(2, 3), \(z) shapiro.test(z)$p.value)
  rejection_rate <- mean(c(tst) < 0.05)
  cat(sprintf("  Shapiro-Wilk rejection rate at alpha=0.05: %.3f (expected ~0.05)\n", rejection_rate))
  stopifnot(abs(rejection_rate - 0.05) < 0.02)

  cat("  PASS\n")
}

test_rnorm_mean_sd <- function() {
  cat("Testing rnorm mean and standard deviation...\n")

  f <- function() {
    nv_rnorm(
      nv_tensor(c(3, 83), dtype = "ui64"),
      dtype = "f64",
      shape_out = c(10L, 10L, 10L, 10L, 10L),
      mu = 10,
      sigma = 9
    )
  }
  g <- jit(f)
  out <- g()
  values <- as_array(out[[2]])

  sample_mean <- mean(values)
  sample_sd <- sd(values)

  cat(sprintf("  Sample mean: %.2f (expected 10)\n", sample_mean))
  cat(sprintf("  Sample SD: %.2f (expected 9)\n", sample_sd))

  stopifnot(abs(sample_mean - 10) < 0.5)
  stopifnot(abs(sample_sd - 9) < 0.5)

  cat("  PASS\n")
}

test_runif_statistical <- function() {
  cat("Testing runif statistical properties...\n")

  f <- function() {
    nv_runif(
      nv_tensor(c(1, 2), dtype = "ui64"),
      dtype = "f32",
      shape_out = c(10, 20, 30, 40, 50),
      lower = -1,
      upper = 1
    )
  }
  g <- jit(f)
  out <- g()
  values <- as_array(out[[2]])

  # Check bounds (should be open interval)
  stopifnot(!any(values == -1))
  stopifnot(!any(values == 1))

  # Check mean (expected 0 for U[-1, 1])
  sample_mean <- mean(values)
  cat(sprintf("  Sample mean: %.5f (expected 0)\n", sample_mean))
  stopifnot(abs(sample_mean) < 1e-3)

  # Check variance (expected 1/3 for U[-1, 1])
  sample_var <- var(values)
  stopifnot(abs(sample_var - 1 / 3) < 1e-3)
}

run_all_tests <- function() {
  cat("Running expensive RNG tests...\n\n")

  test_rnorm_statistical()
  test_rnorm_mean_sd()
  test_runif_statistical()

  cat("\nAll expensive RNG tests passed!\n")
}

run_all_tests()
