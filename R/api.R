# This is the user-facing API containing the exported tensor operations.
#' @include primitives.R

# Special tensor creators

# TODO: We can remove this once we can lift R scalars (nv_broadcast_to(1, c(2, 3, 4)))
nv_constant <- function(value, dtype = NULL, device = NULL, shape) {
  if (length(value) != 1L) {
    stop("value must be a scalar")
  }
  nv_broadcast_to(nv_scalar(value, dtype = dtype, device = device), shape = shape)
}


## Conversion ------------------------------------------------------------------

broadcast_shapes <- function(shape_lhs, shape_rhs) {
  if (length(shape_lhs) > length(shape_rhs)) {
    shape_rhs <- c(rep(1L, length(shape_lhs) - length(shape_rhs)), shape_rhs)
  } else if (length(shape_lhs) < length(shape_rhs)) {
    shape_lhs <- c(rep(1L, length(shape_rhs) - length(shape_lhs)), shape_lhs)
  } else if (identical(shape_lhs, shape_rhs)) {
    return(shape_lhs)
  }
  shape_out <- shape_lhs
  for (i in seq_along(shape_lhs)) {
    d_lhs <- shape_lhs[i]
    d_rhs <- shape_rhs[i]
    if (d_lhs != d_rhs && d_lhs != 1L && d_rhs != 1L) {
      cli_abort("lhs and rhs are not broadcastable")
    }
    shape_out[i] <- max(d_lhs, d_rhs)
  }
  shape_out
}

make_broadcast_dimensions <- function(shape_in, shape_out) {
  rank_in <- length(shape_in)
  rank_out <- length(shape_out)
  if (rank_in == rank_out) {
    # When ranks match, each input dimension maps to the same output dimension
    # StableHLO expects a mapping for every input dim
    return(seq_along(shape_out))
  }
  tail(seq_len(rank_out), rank_in)
}

nv_broadcast_scalar <- function(lhs, rhs) {
  shape_lhs <- shape(lhs)
  shape_rhs <- shape(rhs)
  if (identical(shape_lhs, shape_rhs)) {
    return(list(lhs, rhs))
  }
  if (length(shape_lhs) && length(shape_rhs)) {
    # fmt: skip
    cli_abort(
      "By default, only scalar broadcasting is supported, use {.fn nv_broadcast_tensors} to broadcast higher-dimensional tensors." # nolint
    )
  }
  if (!length(shape_lhs)) {
    lhs <- nv_broadcast_to(lhs, shape_rhs)
  }
  if (!length(shape_rhs)) {
    rhs <- nv_broadcast_to(rhs, shape_lhs)
  }
  list(lhs, rhs)
}

#' @title Broadcast Tensors to a Common Shape
#' @description
#' Broadcast tensors to a common shape.
#'
#' @section Broadcasting Rules:
#' We follow the standard NumPy broadcasting rules:
#' 1. If the tensors have different numbers of dimensions, prepend 1s to the shape of the smaller tensor.
#' 2. For each dimension, if:
#'    - the sizes are the same, do nothing.
#'    - one of the tensors has size 1, expand it to the corresponding size of the other tensor.
#'    - the sizes are different and neither is 1, raise an error.
#'
#' @param ... ([`nv_tensor`])\cr
#'   Tensors to broadcast.
#' @return (`list()` of [`nv_tensor`])
#' @export
nv_broadcast_tensors <- function(...) {
  args <- list(...)
  shape <- Reduce(broadcast_shapes, lapply(args, shape))
  lapply(args, nv_broadcast_to, shape = shape)
}

#' @title Broadcast
#' @description
#' Broadcast a tensor to a given shape using NumPy broadcasting rules.
#' @template param_operand
#' @param shape (`integer()`)\cr
#'   Output shape.
#' @return ([`nv_tensor`])
#' @export
nv_broadcast_to <- function(operand, shape) {
  if (!identical(shape(operand), shape)) {
    broadcast_dimensions <- make_broadcast_dimensions(shape(operand), shape)
    nvl_broadcast_in_dim(operand, shape, broadcast_dimensions)
  } else {
    operand
  }
}

#' @title Convert Tensor to Different Data Type
#' @description
#' Convert a tensor to a different data type.
#' @template param_operand
#' @template param_dtype
#' @return [`nv_tensor`]
#' @export
nv_convert <- function(operand, dtype) {
  nvl_convert(operand, as_dtype(dtype))
}


#' @rdname nv_transpose
#' @export
nv_transpose <- function(x, permutation = NULL) {
  permutation <- permutation %??% rev(seq_len(ndims(x)))
  nvl_transpose(x, permutation)
}


#' @title Reshape
#' @description
#' Reshape a tensor.
#' Note that row-major order is used, which differs from R's column-major order.
#' @template param_operand
#' @param shape (`integer()`)\cr
#'   The new shape.
#' @return [`nv_tensor`]
#' @export
nv_reshape <- nvl_reshape

#' @title Concatenate
#' @description
#' Concatenate a variadic number of tensors.
#' @param ... tensors
#' @param dimension (`integer()`)\cr
#'   The dimension to concatenate along to. Other dimensions must be the same.
#' @return [`nv_tensor`]
#' @export
nv_concatenate <- nvl_concatenate

#' @title Slice
#' @description
#' return slice of operand.
#' @template param_operand
#' @param start_indices start of slice
#' @param limit_indices end of slice
#' @param strides stride size
#' @return [`nv_tensor`]
#' @export
nv_slice <- nvl_slice

#' @title Select
#' @description
#' return values from true_value and false_value conditioned on pred
#' @param pred condition
#' @param true_value on true
#' @param false_value on false
#' @return [`nv_tensor`]
#' @export
nv_select <- nvl_select

## Binary ops ------------------------------------------------------------------

#' @name nv_binary_ops
#' @title Binary Operations
#' @description
#' Binary operations on tensors.
#' @param lhs ([`nv_tensor`])
#' @param rhs ([`nv_tensor`])
#' @return [`nv_tensor`]
NULL

#' @rdname nv_binary_ops
#' @export
nv_add <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_add(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_mul <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_mul(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_sub <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_sub(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_div <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_div(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_pow <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_pow(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_eq <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_eq(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_ne <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_ne(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_gt <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_gt(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_ge <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_ge(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_lt <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_lt(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_le <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_le(args[[1]], args[[2]])
}

## Additional binary ops -------------------------------------------------------

#' @rdname nv_binary_ops
#' @export
nv_max <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_max(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_min <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_min(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_remainder <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_remainder(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_and <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_and(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_or <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_or(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_xor <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_xor(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_shift_left <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_shift_left(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_shift_right_logical <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_shift_right_logical(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_shift_right_arithmetic <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_shift_right_arithmetic(args[[1]], args[[2]])
}

#' @rdname nv_binary_ops
#' @export
nv_atan2 <- function(lhs, rhs) {
  args <- nv_broadcast_scalar(lhs, rhs)
  nvl_atan2(args[[1]], args[[2]])
}


#' @title Bitcast Conversion
#' @name nv_bitcast_convert
#' @description
#' Reinterpret Bits
#' @param operand tensor
#' @param dtype requested dtype
#' @export
nv_bitcast_convert <- nvl_bitcast_convert

## Unary ops ------------------------------------------------------------------

#' @name nv_unary_ops
#' @title Unary Operations
#' @description
#' Unary operations on tensors.
#' @template param_operand
#' @return [`nv_tensor`]

#' @rdname nv_unary_ops
#' @export
nv_neg <- nvl_neg

#' @rdname nv_unary_ops
#' @export
nv_abs <- nvl_abs

#' @rdname nv_unary_ops
#' @export
nv_sqrt <- nvl_sqrt

#' @rdname nv_unary_ops
#' @export
nv_rsqrt <- nvl_rsqrt

#' @rdname nv_unary_ops
#' @export
nv_log <- nvl_log

#' @rdname nv_unary_ops
#' @export
nv_tanh <- nvl_tanh

#' @rdname nv_unary_ops
#' @export
nv_tan <- nvl_tan

#' @rdname nv_unary_ops
#' @export
nv_sine <- nvl_sine

#' @rdname nv_unary_ops
#' @export
nv_cosine <- nvl_cosine

#' @rdname nv_unary_ops
#' @export
nv_floor <- nvl_floor

#' @rdname nv_unary_ops
#' @export
nv_ceil <- nvl_ceil

#' @rdname nv_unary_ops
#' @export
nv_sign <- nvl_sign

#' @rdname nv_unary_ops
#' @export
nv_exp <- nvl_exp

#' @rdname nv_unary_ops
#' @export
#' @param method (`character(1)`)\cr
#'   Method to use for rounding.
#'   Either `"nearest_even"` (default) or `"afz"` (away from zero).
nv_round <- nvl_round

## Other operations -----------------------------------------------------------

#' @title Matrix Multiplication
#' @description
#' Matrix multiplication of two tensors.
#' @section Shapes:
#' - `lhs`: `(b1, ..., bk, m, n)`
#' - `rhs`: `(b1, ..., bk, n, p)`
#' - output: `(b1, ..., bk, m, p)`
#' @section Broadcasting:
#' All dimensions but the last two are broadcasted.
#' @param lhs ([`nv_tensor`])
#' @param rhs ([`nv_tensor`])
#' @return [`nv_tensor`]
#' @export
nv_matmul <- function(lhs, rhs) {
  if (ndims(lhs) < 2L) {
    cli_abort("lhs of matmul must have at least 2 dimensions")
  }
  if (ndims(rhs) < 2L) {
    cli_abort("rhs of matmul must have at least 2 dimensions")
  }
  shape_leading <- broadcast_shapes(head(shape(lhs), -2L), head(shape(rhs), -2L))

  shape_lhs <- c(shape_leading, tail(shape(lhs), 2L))
  shape_rhs <- c(shape_leading, tail(shape(rhs), 2L))

  if (!identical(shape_lhs, shape(lhs))) {
    lhs <- nv_broadcast_to(lhs, shape_lhs)
  }
  if (!identical(shape_rhs, shape(rhs))) {
    rhs <- nv_broadcast_to(rhs, shape_rhs)
  }

  nvl_dot_general(
    lhs,
    rhs,
    contracting_dims = list(ndims(lhs), ndims(rhs) - 1L),
    batching_dims = list(seq_along(shape_leading), seq_along(shape_leading))
  )
}

#' @title Reduction Operators
#' @name nv_reduce_ops
#' @description
#' Reduce a tensor along specified dimensions.
#' @template param_operand
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions.
#' @return [`nv_tensor`]
#' @export
nv_reduce_sum <- nvl_reduce_sum

#' @rdname nv_reduce_ops
#' @export
nv_reduce_mean <- function(operand, dims, drop = TRUE) {
  # TODO: division by zero?
  nelts <- prod(shape(operand)[dims])
  # TODO: Should just be able to do use autocasting and divide by nelts scalar
  nv_reduce_sum(operand, dims, drop) / nv_scalar(nelts, dtype(operand))
}

#' @rdname nv_reduce_ops
#' @export
nv_reduce_prod <- nvl_reduce_prod

#' @rdname nv_reduce_ops
#' @export
nv_reduce_max <- nvl_reduce_max

#' @rdname nv_reduce_ops
#' @export
nv_reduce_min <- nvl_reduce_min

#' @rdname nv_reduce_ops
#' @export
nv_reduce_any <- nvl_reduce_any

#' @rdname nv_reduce_ops
#' @export
nv_reduce_all <- nvl_reduce_all

#' @title Random Numbers
#' @name nv_rng_bit_generator
#' @description
#' generate random bits of desired shape and dtype
#' @param initial_state state seed
#' @param rng_algorithm one of 'DEFAULT', 'THREE_FRY', 'PHILOX'
#' @param dtype datatype of output
#' @param shape_out output shape
#' @export
nv_rng_bit_generator <- nvl_rng_bit_generator

#' @title Internal: Random Unit Uniform Numbers
#' @name nv_runif
#' @description
#' generate random uniform numbers in [0, 1)
#' @param initial_state state seed
#' @param dtype output dtype either "f32" or "f64"
#' @param shape_out output shape
nv_unif_rand <- function(
  initial_state,
  dtype = "f64",
  shape_out
) {
  checkmate::assertChoice(dtype, c("f32", "f64"))
  checkmate::assertIntegerish(shape_out, lower = 1, min.len = 1, any.missing = FALSE)

  # generate random bits
  # use THREE_FRY as rng algorithm: JAX default
  rbits <- nv_rng_bit_generator(
    initial_state = initial_state,
    "THREE_FRY",
    paste0("ui", sub("f(\\d+)", "\\1", dtype)),
    shape_out = shape_out
  )

  # shift value: 9 for f32, 11 for f64
  shift <- nv_scalar(
    ifelse(dtype == "f32", 9L, 11L),
    dtype = paste0("ui", sub("f(\\d+)", "\\1", dtype))
  )

  # shift to the right, s.t. exponent bits are all 0
  mantissa <- nv_shift_right_logical(rbits[[2]], shift)

  # interpretation of 1.0 (float) as unsigned
  one_bits <- nv_bitcast_convert(
    nv_scalar(1.0, dtype = dtype),
    dtype = paste0("ui", sub("f(\\d+)", "\\1", dtype))
  )

  # bitwise or -> exponent from 1.0 (float), mantissa is random
  U <- nv_or(mantissa, one_bits)

  # convert back to requested dtype
  # resulting RVs  are in [1, 2)
  U <- nv_bitcast_convert(U, dtype = dtype)

  # shift to [0, 1)
  U <- nv_add(U, nv_scalar(-1, dtype = dtype))

  # return state and RVs
  list(rbits[[1]], U)
}


#' @title Random Uniform Numbers
#' @name nv_runif
#' @description
#' generate random uniform numbers in ]lower, upper[
#' @param initial_state state seed
#' @param dtype output dtype either "f32" or "f64"
#' @param shape_out output shape
#' @param lower lower bound
#' @param upper upper bound
#' @export
nv_runif <- function(
  initial_state,
  dtype = "f64",
  shape_out,
  lower = 0,
  upper = 1
) {
  checkmate::assertChoice(dtype, c("f32", "f64"))
  checkmate::assertNumeric(lower, len = 1, any.missing = FALSE, upper = upper)
  checkmate::assertNumeric(upper, len = 1, any.missing = FALSE, lower = lower)
  checkmate::assertIntegerish(shape_out, lower = 1, min.len = 1, any.missing = FALSE)

  if (upper == lower) {
    return(nv_broadcast_to(nv_scalar(upper, dtype = dtype), shape = shape_out))
  }

  .lower <- nv_scalar(lower, dtype = dtype)
  .upper <- nv_scalar(upper, dtype = dtype)
  .range <- nv_sub(.upper, .lower)

  # generate samples in [0, 1)
  Unif <- nv_unif_rand(initial_state = initial_state, shape_out = shape_out, dtype = dtype)
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
    shape = shape_out
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
#' @name nv_rnorm
#' @description
#' generate random normal numbers
#' @param initial_state state seed
#' @param dtype output dtype either "f32" or "f64"
#' @param shape_out output shape
#' @param mu scalar: expected value
#' @param sigma scalar: standard deviation
#' #' @section Covariance:
#' To implement a covariance structure use cholesky decomposition
#' @export
nv_rnorm <- function(initial_state, dtype, shape_out, mu = 0, sigma = 1) {
  checkmate::assertChoice(dtype, c("f32", "f64"))
  checkmate::assertNumeric(mu, len = 1, any.missing = FALSE)
  checkmate::assertNumeric(sigma, len = 1, any.missing = FALSE, lower = 0)
  checkmate::assertIntegerish(shape_out, lower = 1, min.len = 1, any.missing = FALSE)
  # n: amount of rvs needed
  n <- prod(shape_out)

  # Box-Muller Method:
  # from to random uniform variables u1 and u2 we can produce to normals z1, z2
  # z1 = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
  # z2 = sqrt(-2 * log(u1)) * sin(2 * pi * u2)
  # Box-Muller works via polar representation of coordinates.
  # We scale this approach and genereate ceil(n/2) uniform rvs twice (U, Theta)

  # generate the first ceil(n/2) random uniform variables
  U <- nv_unif_rand(
    initial_state = initial_state,
    dtype = dtype,
    shape_out = as.integer(ceiling(n / 2))
  )

  # compute the radius R = sqrt(-2 * log(u1))
  R <- nv_mul(nv_log(U[[2]]), nv_scalar(-2, dtype = dtype))
  sqrt_R <- nv_sqrt(R)

  # generate second batch of ceil(n/2) random uniform variables
  Theta <- nv_unif_rand(initial_state = U[[1]], dtype = dtype, shape_out = c(ceiling(n / 2)))

  # compute cos(2 * pi * u2) / sin(2 * pi * u2)
  Theta[[2]] <- nv_mul(Theta[[2]], nv_scalar(2 * pi, dtype = dtype))
  sin_Theta <- nv_sine(Theta[[2]])
  cos_Theta <- nv_cosine(Theta[[2]])

  # compue z1, z2
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

  # if n is uneven, only keep N(1,...,n-1), i.e. discard last entry of N
  if (n %% 2 == 1) {
    N <- nv_slice(N, start_indices = 1L, limit_indices = as.integer(n + 1), strides = 1L)
  }

  # reshape N to match requested shape
  N <- nv_reshape(N, shape = shape_out)

  # return state and Normals N
  list(Theta[[1]], N)
}

## Data Types ------------------------------------------------------------------

#' @title Tensor Data Types
#' @name data_types
#' @description
#' Data types for tensors:
#' - `dt_i1`: boolean.
#' - `dt_i{8, 16, 32, 64}`: signed integer.
#' - `dt_ui{8, 16, 32, 64}`: unsigned integer.
#' - `dt_f{32, 64}`: float.
NULL

#' @rdname data_types
#' @format NULL
#' @usage NULL
#' @export
dt_i1 <- BooleanType()
#' @rdname data_types
#' @format NULL
#' @usage NULL
#' @format NULL
#' @usage NULL
#' @export
dt_i8 <- IntegerType(8)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_i16 <- IntegerType(16)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_i32 <- IntegerType(32)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_i64 <- IntegerType(64)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_ui8 <- UnsignedType(8)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_ui16 <- UnsignedType(16)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_ui32 <- UnsignedType(32)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_ui64 <- UnsignedType(64)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_f32 <- FloatType(32)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_f64 <- FloatType(64)


# Higher order primitives

#' @title If
#' @description
#' Functional if statement.
#' @param pred ([`nv_tensor`])\cr
#'   Flag.
#' @param true (NSE)\cr
#'   Expression to evaluate if the condition is true.
#' @param false (NSE)\cr
#'   Expression to evaluate if the condition is false.
#' @return [`nv_tensor`]
#' @export
nv_if <- nvl_if

#' @title While
#' @description
#' Functional while loop.
#' @param init (`list()`)\cr
#'   Initial state.
#' @param cond (`function`)\cr
#'   Condition function: `f: state -> bool`.
#' @param body (`function`)\cr
#'   Body function. `f: state -> state`.
#' @return [`nv_tensor`]
#' @export
nv_while <- nvl_while
