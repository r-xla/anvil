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
  dtype = "f32",
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
nv_rnorm <- function(initial_state, dtype = "f32", shape, mu = 0, sigma = 1) {
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

  n <- prod(shape)

  # We generate ui8 random values and extract 8 bits from each.
  n_bytes <- as.integer(ceiling(n / 8))

  # Generating i1 is not possible unfortunately
  rbits <- nvl_rng_bit_generator(
    initial_state = initial_state,
    "THREE_FRY",
    dtype = "ui8",
    shape_out = n_bytes
  )

  # Shift the 8 bits by 0, 1, 2, and xor with 1 (u8) to extract individual bits
  bytes_col <- nv_reshape(rbits[[2]], shape = c(n_bytes, 1L))
  bit_positions <- nv_iota(dim = 2L, shape = c(1L, 8L), dtype = "ui8", start = 0)
  bc <- nv_broadcast_tensors(bytes_col, bit_positions) # (n_bytes, 8)
  bytes_bc <- bc[[1L]]
  positions_bc <- bc[[2L]]
  shifted <- nvl_shift_right_logical(bytes_bc, positions_bc) # (n_bytes, 8)

  one <- nv_scalar(1L, dtype = "ui8")
  bits <- nv_and(shifted, one)

  bits <- nv_reshape(bits, shape = n_bytes * 8L)

  # discard unneeded bits
  if (n < n_bytes * 8L) {
    bits <- nv_slice(bits, start_indices = 1L, limit_indices = n, strides = 1L)
  }

  bits <- nv_reshape(bits, shape = shape)
  bits <- nv_convert(bits, dtype = dtype)
  list(rbits[[1]], bits)
}

#' @title Random Discrete Sample
#' @description
#' Sample from a discrete distribution, analogous to R's `sample()` function.
#' Samples integers from 1 to n with uniform probability and with replacement.
#' @param n (`integer(1)`)\cr
#'   Number of categories to sample from (samples integers 1 to n).
#' @template param_shape
#' @param initial_state ([`tensorish`])\cr
#'   Tensor of type `ui64[2]` for the RNG state.
#' @param dtype (`character(1)` | [`TensorDataType`])\cr
#'   Output dtype (default "i32").
#' @return (`list()` of [`tensorish`])\cr
#'   List of two tensors: the new RNG state and the sampled integers (1 to n).
#' @export
nv_sample_int <- function(initial_state, n, shape, dtype = "i32") {
  checkmate::assert_int(n, lower = 1)
  shape <- assert_shapevec(shape)
  n_sample <- prod(shape)

  # we sample uniformly and compute the maximial i, s.t. sum(bits[1:i]) <= F(x)

  # use f64 for higher precision
  res <- nv_unif_rand(initial_state, shape = n_sample, dtype = "f64")
  u <- res[[2]]

  cp <- nv_div(
    nv_iota(dim = 1L, shape = n, dtype = "f64", start = 1),
    nv_fill(n, "f64", shape = c())
  )

  u_col <- nv_reshape(u, c(n_sample, 1L))
  cp_row <- nv_reshape(cp, c(1L, n))
  bc <- nv_broadcast_tensors(u_col, cp_row) # (n_sample, n)
  lt_matrix <- nv_convert(nv_lt(bc[[2L]], bc[[1L]]), dtype = "i32")
  samples <- nv_add(nv_reduce_sum(lt_matrix, dims = 2L), 1L)

  return(list(res[[1]], nv_convert(nv_reshape(samples, shape), dtype)))
}
