# Expensive RNG tests for statistical verification
# They are slow, so we don't run them in the normal CI suite

library(anvil)

test_rnorm_statistical <- function() {
  cat("Testing rnorm statistical properties...\n")

  # Test normality with large sample
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f64", shape = c(200L, 300L, 400L))
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
      shape = c(10L, 10L, 10L, 10L, 10L),
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
      shape = c(10, 20, 30, 40, 50),
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

  cat("  PASS\n")
}

test_rbinom_statistical <- function() {
  cat("Testing nv_rbinom statistical properties...\n")

  # Generate a large sample of Binomial(1, 0.5) samples (i.e., Bernoulli(0.5))
  f <- function() {
    nv_rbinom(
      nv_tensor(c(1, 2), dtype = "ui64"),
      dtype = "i32",
      shape = c(100L, 100L, 100L)  # 1 million samples
    )
  }
  g <- jit(f)
  out <- g()
  values <- as_array(out[[2]])

  # All values should be 0 or 1
  stopifnot(all(values %in% c(0L, 1L)))

  # Check proportion of 1s (expected 0.5 for fair coin)
  prop_ones <- mean(values)
  cat(sprintf("  Proportion of 1s: %.5f (expected 0.5)\n", prop_ones))
  # With 1M samples, we expect prop to be within ~0.002 of 0.5 with high probability
  stopifnot(abs(prop_ones - 0.5) < 0.005)

  # Check variance (expected 0.25 for Binomial(1, 0.5))
  sample_var <- var(c(values))
  cat(sprintf("  Sample variance: %.5f (expected 0.25)\n", sample_var))
  stopifnot(abs(sample_var - 0.25) < 0.005)

  # Run a binomial test on smaller chunks to check uniformity
  # Split into 1000 chunks of 1000 samples each
  chunk_sums <- apply(array(values, dim = c(1000, 1000)), 2, sum)
  # Each chunk sum should follow Binomial(1000, 0.5)
  # Expected mean = 500, expected sd = sqrt(250) ≈ 15.8

  # Check that chunk means are reasonable (within 3 sd of expected)
  chunk_mean <- mean(chunk_sums)
  cat(sprintf("  Mean of chunk sums: %.2f (expected 500)\n", chunk_mean))
  stopifnot(abs(chunk_mean - 500) < 5)

  # Chi-squared test for uniformity of bits
  # We expect roughly equal number of 0s and 1s
  n_ones <- sum(values)
  n_zeros <- length(values) - n_ones
  chi_sq <- (n_ones - n_zeros)^2 / length(values)
  cat(sprintf("  Chi-squared statistic: %.4f (should be small)\n", chi_sq))
  # Chi-squared with df=1 at alpha=0.01 is 6.635
  stopifnot(chi_sq < 6.635)

  cat("  PASS\n")
}

test_sample_int_statistical <- function() {
  cat("Testing nv_sample_int statistical properties...\n")

  # Test 1: Equal probabilities (uniform discrete)
  cat("  Testing equal probabilities...\n")
  f1 <- function() {
    nv_sample_int(
      n = 6L,
      shape = 60000L,
      initial_state = nv_tensor(c(1, 2), dtype = "ui64")
    )
  }
  g1 <- jit(f1)
  out1 <- g1()
  values1 <- as_array(out1[[2]])

  # All values should be in 1:6
  stopifnot(all(values1 >= 1L & values1 <= 6L))

  # Check that each category appears roughly 1/6 of the time
  for (i in 1:6) {
    prop <- mean(values1 == i)
    cat(sprintf("    Category %d: %.4f (expected ~0.1667)\n", i, prop))
    stopifnot(abs(prop - 1 / 6) < 0.01)
  }

  cat("  PASS\n")
}

run_all_tests <- function() {
  cat("Running expensive RNG tests...\n\n")

  test_rnorm_statistical()
  test_rnorm_mean_sd()
  test_runif_statistical()
  test_rbinom_statistical()
  test_sample_int_statistical()

  cat("\nAll expensive RNG tests passed!\n")
}

run_all_tests()
