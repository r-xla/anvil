# Random Number Generation API
# This file contains user-facing RNG sampling functions

#' @title Random Uniform Numbers
#' @description
#' generate random uniform numbers in ]lower, upper[
#' @param initial_state ([`tensorish`])\cr
#'   Tensor of type `ui64[2]`.
#' @template param_dtype
#' @template param_shape
#' @param lower,upper (`numeric(1)`)\cr
#'   Lower and upper bound.
#' @return (`list()` of [`tensorish`])\cr
#'   List of two tensors: the new RNG state and the generated random numbers.
#' @export
nv_runif <- function(
  initial_state,
  dtype = "f64",
  shape,
  lower = 0,
  upper = 1
) {
  checkmate::assertChoice(dtype, c("f32", "f64"))
  checkmate::assertNumeric(lower, len = 1, any.missing = FALSE, upper = upper)
  checkmate::assertNumeric(upper, len = 1, any.missing = FALSE, lower = lower)
  shape <- assert_shapevec(shape)

  if (upper == lower) {
    return(nv_broadcast_to(nv_scalar(upper, dtype = dtype), shape = shape))
  }

  .lower <- nv_scalar(lower, dtype = dtype)
  .upper <- nv_scalar(upper, dtype = dtype)
  .range <- nv_sub(.upper, .lower)

  # generate samples in [0, 1)
  Unif <- nv_unif_rand(initial_state = initial_state, shape = shape, dtype = dtype)
  U <- Unif[[2]]

  # check if some values are <= 0
  le_zero <- nv_le(U, nv_scalar(0, dtype = dtype))

  # Define smallest step (like R's 0.5 * i2_32m1 philosophy)
  # for f32 and 23 mantissa bits 2^-24 lies between 0 and 2^-23,
  # the next smallest generated value.
  # Same applies for f64 and 2^-53 and 52 mantissa bits.
  smallest_step <- nv_broadcast_to(
    nv_scalar(
      ifelse(dtype == "f32", 2^-24, 2^-53),
      dtype = dtype
    ),
    shape = shape
  )

  # Replace values <= 0 with smallest_step
  U <- nv_select(le_zero, smallest_step, U)

  # expand to range
  U <- nv_mul(U, .range)
  # shift to interval
  U <- nv_add(U, .lower)

  return(list(Unif[[1]], U))
}

#' @title Random Normal Numbers
#' @description
#' generate random normal numbers
#' @param initial_state ([`tensorish`])\cr
#'   Tensor of type `ui64[2]`.
#' @template param_dtype
#' @template param_shape
#' @param mu (`numeric(1)`)\cr
#'   Expected value.
#' @param sigma (`numeric(1)`)\cr
#'   Standard deviation.
#' #' @section Covariance:
#' To implement a covariance structure use cholesky decomposition.
#' @return (`list()` of [`tensorish`])\cr
#'   List of two tensors: the new RNG state and the generated random numbers.
#' @export
nv_rnorm <- function(initial_state, dtype, shape, mu = 0, sigma = 1) {
  checkmate::assertChoice(dtype, c("f32", "f64"))
  checkmate::assertNumeric(mu, len = 1, any.missing = FALSE)
  checkmate::assertNumeric(sigma, len = 1, any.missing = FALSE, lower = 0)
  shape <- assert_shapevec(shape)
  # n: amount of rvs needed
  n <- prod(shape)

  # Box-Muller Method:
  # from two random uniform variables u1 and u2 we can produce to normals z1, z2
  # z1 = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
  # z2 = sqrt(-2 * log(u1)) * sin(2 * pi * u2)
  # Box-Muller works via polar representation of coordinates.
  # We scale this approach and genereate ceil(n/2) uniform rvs twice (U, Theta)

  # generate the first ceil(n/2) random uniform variables
  U <- nv_unif_rand(
    initial_state = initial_state,
    dtype = dtype,
    shape = as.integer(ceiling(n / 2))
  )

  # compute the radius R = sqrt(-2 * log(u1))
  R <- nv_mul(nv_log(U[[2]]), nv_scalar(-2, dtype = dtype))
  sqrt_R <- nv_sqrt(R)

  # generate second batch of ceil(n/2) random uniform variables
  Theta <- nv_unif_rand(initial_state = U[[1]], dtype = dtype, shape = as.integer(ceiling(n / 2)))

  # compute cos(2 * pi * u2) / sin(2 * pi * u2)
  Theta[[2]] <- nv_mul(Theta[[2]], nv_scalar(2 * pi, dtype = dtype))
  sin_Theta <- nv_sine(Theta[[2]])
  cos_Theta <- nv_cosine(Theta[[2]])

  # compute z1, z2
  Z1 <- nv_mul(sqrt_R, sin_Theta)
  Z2 <- nv_mul(sqrt_R, cos_Theta)

  # concatenate z = (z1, z2)
  Z <- nv_concatenate(Z1, Z2, dimension = 1L)

  # multiply with requested sd:
  # was:    var(Z) = 1
  # now:    var(Z) = sd^2
  N <- nv_mul(Z, nv_scalar(sigma, dtype = dtype))

  # add requested mu:
  # was:    mean(Z) = 0
  # now:    mean(Z) = mu
  N <- nv_add(N, nv_scalar(mu, dtype = dtype))

  # if n is uneven, only keep N(1,...,n), i.e. discard last entry of N
  if (n %% 2 == 1) {
    N <- nv_slice(N, start_indices = 1L, limit_indices = n, strides = 1L)
  }

  # reshape N to match requested shape
  N <- nv_reshape(N, shape = shape)

  # return state and Normals N
  list(Theta[[1]], N)
}

#' @title Random Binomial Samples
#' @description
#' Generate random Binomial(1, 0.5) samples (0 or 1) by extracting individual bits
#' from the random number generator. This is equivalent to Bernoulli(0.5) samples.
#' @param initial_state ([`tensorish`])\cr
#'   Tensor of type `ui64[2]`.
#' @template param_dtype
#' @template param_shape
#' @return (`list()` of [`tensorish`])\cr
#'   List of two tensors: the new RNG state and the generated random samples (0 or 1).
#' @export
nv_rbinom <- function(initial_state, dtype = "i32", shape) {
  shape <- assert_shapevec(shape)

  # Total number of samples needed
  n <- prod(shape)

  # We generate ui8 random values and extract 8 bits from each.
  # Generating i1 is not possible unfortunately
  n_bytes <- as.integer(ceiling(n / 8))

  # Generate random bytes
  rbits <- nvl_rng_bit_generator(
    initial_state = initial_state,
    "THREE_FRY",
    dtype = "ui8",
    shape_out = n_bytes
  )

  # Extract individual bits from each byte
  # bits[i, j] = (rbits[i] >> j) & 1 for j in 0:7
  # Reshape rbits to (n_bytes, 1) for broadcasting
  bytes_col <- nv_reshape(rbits[[2]], shape = c(n_bytes, 1L))

  # Create bit positions [0, 1, 2, 3, 4, 5, 6, 7] as ui8
  bit_positions <- nv_tensor(0:7, dtype = "ui8", shape = c(1L, 8L))

  # Broadcast both tensors to (n_bytes, 8) for element-wise shift
  bc <- nv_broadcast_tensors(bytes_col, bit_positions)
  bytes_bc <- bc[[1L]]
  positions_bc <- bc[[2L]]

  # Shift each byte right by bit positions (result is (n_bytes, 8))
  shifted <- nvl_shift_right_logical(bytes_bc, positions_bc)

  # AND with 1 to extract individual bits
  one <- nv_scalar(1L, dtype = "ui8")
  bits <- nv_and(shifted, one)

  # Flatten to 1D: (n_bytes * 8,)
  bits <- nv_reshape(bits, shape = n_bytes * 8L)

  # Slice to get exactly n bits (discard extra bits if n is not a multiple of 8)
  if (n < n_bytes * 8L) {
    bits <- nv_slice(bits, start_indices = 1L, limit_indices = n, strides = 1L)
  }

  # Reshape to the desired output shape
  bits <- nv_reshape(bits, shape = shape)

  # Convert to the desired output dtype
  bits <- nv_convert(bits, dtype = dtype)

  # Return state and samples
  list(rbits[[1]], bits)
}

#' @title Random Discrete Sample
#' @description
#' Sample from a discrete distribution, analogous to R's `sample()` function.
#' Samples integers from 1 to n with optional probability weights.
#' @param n (`integer(1)`)\cr
#'   Number of categories to sample from (samples integers 1 to n).
#' @template param_shape
#' @param replace (`logical(1)`)\cr
#'   Should sampling be with replacement? Default is `TRUE`.
#' @param prob ([`tensorish`] | `NULL`)\cr
#'   A tensor of probability weights of length n. If `NULL` (default),
#'   equal probabilities are used. Probabilities will be normalized to sum to 1.
#' @param initial_state ([`tensorish`])\cr
#'   Tensor of type `ui64[2]` for the RNG state.
#' @param dtype (`character(1)` | [`TensorDataType`])\cr
#'   Output dtype (default "i32").
#' @return (`list()` of [`tensorish`])\cr
#'   List of two tensors: the new RNG state and the sampled integers (1 to n).
#' @export
nv_rdiscrete <- function(n, shape, replace = TRUE, prob = NULL, initial_state, dtype = "i32") {
  checkmate::assert_int(n, lower = 1)
  shape <- assert_shapevec(shape)
  k <- prod(shape) # number of samples

  if (!replace) {
    # Sampling without replacement using sequential algorithm
    # Based on R's SampleNoReplace: for each sample, pick from remaining
    # elements proportional to their (remaining) probabilities
    if (k > n) {
      cli_abort("Cannot take {k} samples without replacement from {n} categories")
    }

    # Pre-generate k uniform random numbers
    U <- nv_unif_rand(initial_state = initial_state, dtype = "f64", shape = k)
    new_state <- U[[1L]]
    u_samples <- U[[2L]]

    # Normalize probabilities (or use uniform 1/n)
    if (is.null(prob)) {
      probs <- nv_tensor(rep(1.0 / n, n), dtype = "f64", shape = n)
    } else {
      prob_sum <- nv_reduce_sum(prob, dims = 1L, drop = TRUE)
      probs <- nv_div(prob, prob_sum)
    }

    # Lower triangular matrix for cumsum: lower_tri[i,j] = 1 if i >= j
    row_idx <- nv_tensor(rep(seq_len(n), times = n), dtype = "i32", shape = c(n, n))
    col_idx <- nv_tensor(rep(seq_len(n), each = n), dtype = "i32", shape = c(n, n))
    lower_tri <- nv_convert(nv_ge(row_idx, col_idx), dtype = "f64")

    # Indices for one-hot creation (0-based for comparison)
    indices_0based <- nv_tensor(seq_len(n) - 1L, dtype = "i32", shape = n)
    # Indices for output (1-based)
    indices_1based <- nv_tensor(seq_len(n), dtype = "f64", shape = n)
    # Sample position indices (0-based)
    sample_indices <- nv_tensor(seq_len(k) - 1L, dtype = "i32", shape = k)

    # Initial state for while loop
    init <- list(
      mask = nv_tensor(rep(1.0, n), dtype = "f64", shape = n), # 1 = available
      samples = nv_tensor(rep(0.0, k), dtype = "f64", shape = k),
      i = nv_scalar(0L, dtype = "i32")
    )

    # Condition: i < k (receives state elements as separate arguments)
    cond <- function(mask, samples, i) {
      nv_lt(i, nv_scalar(k, dtype = "i32"))
    }

    # Body: sample one element without replacement (receives state elements as separate arguments)
    body <- function(mask, samples, i) {
      # Masked probabilities (only available elements)
      masked_probs <- nv_mul(probs, mask)

      # Remaining total mass
      totalmass <- nv_reduce_sum(masked_probs, dims = 1L, drop = TRUE)

      # Get u[i] and compute target = u[i] * totalmass
      # Create one-hot for position i to extract u[i]
      i_onehot <- nv_convert(nv_eq(sample_indices, i), "f64")
      u_i <- nv_reduce_sum(nv_mul(u_samples, i_onehot), dims = 1L, drop = TRUE)
      target <- nv_mul(u_i, totalmass)

      # Compute cumsum of masked_probs via matrix multiplication
      probs_col <- nv_reshape(masked_probs, c(n, 1L))
      cumsum_col <- nv_matmul(lower_tri, probs_col)
      cumsum <- nv_reshape(cumsum_col, shape = n)

      # Find first index where cumsum >= target
      # selected_idx = count of positions where cumsum < target
      lt_mask <- nv_convert(nv_lt(cumsum, target), "i32")
      selected_idx <- nv_reduce_sum(lt_mask, dims = 1L, drop = TRUE) # 0-based

      # Get the 1-based index value at selected position
      selected_onehot <- nv_convert(nv_eq(indices_0based, selected_idx), "f64")
      selected_value <- nv_reduce_sum(nv_mul(indices_1based, selected_onehot), dims = 1L, drop = TRUE)

      # Update samples: samples[i] = selected_value
      # samples_new = samples + selected_value * one_hot(i)
      new_samples <- nv_add(samples, nv_mul(selected_value, i_onehot))

      # Update mask: mask[selected_idx] = 0
      # mask_new = mask * (1 - one_hot(selected_idx))
      new_mask <- nv_mul(mask, nv_sub(nv_scalar(1.0, "f64"), selected_onehot))

      # Increment i
      new_i <- nv_add(i, nv_scalar(1L, dtype = "i32"))

      list(mask = new_mask, samples = new_samples, i = new_i)
    }

    # Run the while loop
    result <- nv_while(init, cond, body)
    samples <- result$samples

    samples <- nv_convert(samples, dtype = dtype)
    if (!identical(shape, k)) {
      samples <- nv_reshape(samples, shape = shape)
    }
    return(list(new_state, samples))
  }

  # Sampling with replacement
  # Generate uniform samples in [0, 1)
  U <- nv_unif_rand(initial_state = initial_state, dtype = "f64", shape = k)
  new_state <- U[[1L]]
  u_samples <- U[[2L]]

  if (is.null(prob)) {
    # Equal probabilities: sample = floor(u * n) + 1
    samples <- nv_add(
      nv_floor(nv_mul(u_samples, nv_scalar(as.double(n), dtype = "f64"))),
      nv_scalar(1, dtype = "f64")
    )
  } else {
    # Custom probabilities: use inverse CDF method
    prob_sum <- nv_reduce_sum(prob, dims = 1L, drop = TRUE)
    prob_normalized <- nv_div(prob, prob_sum)

    # Compute cumsum via lower triangular matrix multiplication
    row_idx <- nv_tensor(rep(seq_len(n), times = n), dtype = "i32", shape = c(n, n))
    col_idx <- nv_tensor(rep(seq_len(n), each = n), dtype = "i32", shape = c(n, n))
    lower_tri <- nv_convert(nv_ge(row_idx, col_idx), dtype = "f64")

    prob_col <- nv_reshape(prob_normalized, shape = c(n, 1L))
    cumsum_col <- nv_matmul(lower_tri, prob_col)
    cumsum <- nv_reshape(cumsum_col, shape = n)

    # Create cumsum_exclusive: [0, cumsum[1], ..., cumsum[n-1]]
    zero <- nv_tensor(0, dtype = "f64", shape = 1L)
    if (n > 1L) {
      cumsum_head <- nv_slice(cumsum, start_indices = 1L, limit_indices = n - 1L, strides = 1L)
      cumsum_exclusive <- nv_concatenate(zero, cumsum_head, dimension = 1L)
    } else {
      cumsum_exclusive <- zero
    }

    # For each u, count bins where u >= cumsum_exclusive (gives category 1 to n)
    u_col <- nv_reshape(u_samples, shape = c(k, 1L))
    cumsum_row <- nv_reshape(cumsum_exclusive, shape = c(1L, n))
    bc <- nv_broadcast_tensors(u_col, cumsum_row)
    ge_matrix <- nv_convert(nvl_ge(bc[[1L]], bc[[2L]]), dtype = "f64")
    samples <- nv_reduce_sum(ge_matrix, dims = 2L, drop = TRUE)
  }

  # Convert to output dtype
  samples <- nv_convert(samples, dtype = dtype)

  # Reshape to requested shape
  if (!identical(shape, k)) {
    samples <- nv_reshape(samples, shape = shape)
  }

  list(new_state, samples)
}
