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

test_rdiscrete_statistical <- function() {
  cat("Testing nv_rdiscrete statistical properties...\n")

  # Test 1: Equal probabilities (uniform discrete)
  cat("  Testing equal probabilities...\n")
  f1 <- function() {
    nv_rdiscrete(
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
    stopifnot(abs(prop - 1/6) < 0.01)
  }

  # Test 2: Custom probabilities
  cat("  Testing custom probabilities [0.1, 0.3, 0.6]...\n")
  expected_probs <- c(0.1, 0.3, 0.6)
  prob <- nv_tensor(expected_probs, dtype = "f64")

  f2 <- function(p) {
    nv_rdiscrete(
      n = 3L,
      shape = 100000L,
      prob = p,
      initial_state = nv_tensor(c(3, 4), dtype = "ui64")
    )
  }
  g2 <- jit(f2)
  out2 <- g2(prob)
  values2 <- as_array(out2[[2]])

  # Check proportions match expected
  for (i in 1:3) {
    prop <- mean(values2 == i)
    expected <- expected_probs[i]
    cat(sprintf("    Category %d: %.4f (expected %.2f)\n", i, prop, expected))
    stopifnot(abs(prop - expected) < 0.01)
  }

  # Test 3: Chi-squared goodness of fit
  cat("  Chi-squared test...\n")
  observed <- table(factor(values2, levels = 1:3))
  expected_counts <- length(values2) * expected_probs
  chi_sq <- sum((observed - expected_counts)^2 / expected_counts)
  cat(sprintf("    Chi-squared: %.4f (df=2, critical value at 0.01 is 9.21)\n", chi_sq))
  stopifnot(chi_sq < 9.21)

  cat("  PASS\n")
}

test_rdiscrete_without_replacement <- function() {
  cat("Testing nv_rdiscrete with replace = FALSE...\n")

  # Test 1: All samples should be unique
  cat("  Test uniqueness...\n")
  f1 <- function(s) {
    nv_rdiscrete(n = 100L, shape = 50L, replace = FALSE, initial_state = s)
  }
  g1 <- jit(f1)

  # Run multiple trials
  state <- nv_tensor(c(1, 2), dtype = "ui64")
  for (trial in 1:10) {
    out <- g1(state)
    values <- c(as_array(out[[2]]))
    stopifnot(length(unique(values)) == 50L)
    stopifnot(all(values >= 1L & values <= 100L))
    state <- out[[1]]  # update state
  }
  cat("    All 10 trials produced unique samples\n")

  # Test 2: Permutation test (k = n should give a permutation)
  cat("  Test full permutation...\n")
  f2 <- function(s) {
    nv_rdiscrete(n = 20L, shape = 20L, replace = FALSE, initial_state = s)
  }
  g2 <- jit(f2)

  state <- nv_tensor(c(3, 5), dtype = "ui64")
  for (trial in 1:5) {
    out <- g2(state)
    values <- c(as_array(out[[2]]))
    stopifnot(identical(sort(values), 1:20))
    state <- out[[1]]
  }
  cat("    All 5 trials produced valid permutations\n")

  # Test 3: Weighted sampling without replacement
  cat("  Test weighted sampling without replacement...\n")
  # Higher weights should be selected more often across many trials
  probs <- c(0.01, 0.01, 0.01, 0.01, 0.96)  # strongly favor item 5
  prob_tensor <- nv_tensor(probs, dtype = "f64")

  f3 <- function(s, p) {
    nv_rdiscrete(n = 5L, shape = 3L, replace = FALSE, prob = p, initial_state = s)
  }
  g3 <- jit(f3)

  # Count how often each item appears in samples across trials
  counts <- rep(0L, 5)
  state <- nv_tensor(c(7, 11), dtype = "ui64")
  n_trials <- 1000L

  for (trial in seq_len(n_trials)) {
    out <- g3(state, prob_tensor)
    values <- c(as_array(out[[2]]))
    stopifnot(length(unique(values)) == 3L)
    for (v in values) counts[v] <- counts[v] + 1L
    state <- out[[1]]
  }

  # Item 5 (with 0.96 probability) should almost always be selected
  prop_5 <- counts[5] / n_trials
  cat(sprintf("    Item 5 (prob=0.96) selected in %.1f%% of trials\n", prop_5 * 100))
  stopifnot(prop_5 > 0.95)  # should be selected almost always

  # Low probability items should be selected less often
  prop_low <- mean(counts[1:4]) / n_trials
  cat(sprintf("    Low prob items (0.01 each) avg selection rate: %.1f%%\n", prop_low * 100))
  # Expected: each low-prob item selected in roughly 50% of trials (picking 3 from 5, with item 5 almost always included)
  stopifnot(prop_low < 0.7)

  cat("  PASS\n")
}

run_all_tests <- function() {
  cat("Running expensive RNG tests...\n\n")

  test_rnorm_statistical()
  test_rnorm_mean_sd()
  test_runif_statistical()
  test_rbinom_statistical()
  test_rdiscrete_statistical()
  test_rdiscrete_without_replacement()

  cat("\nAll expensive RNG tests passed!\n")
}

run_all_tests()
