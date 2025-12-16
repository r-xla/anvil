# This is the user-facing API containing the exported tensor operations.
#' @include primitives.R

# Special tensor creators

#' @title Constant
#' @description
#' Create a constant.
#' @param value (any)\cr
#'   Value.
#' @param shape (integer())\cr
#'   Shape.
#' @param dtype (character(1))\cr
#'   Data type.
#' @export
nv_fill <- function(value, shape, dtype = NULL) {
  dtype <- if (is.null(dtype)) {
    default_dtype(value)
  } else {
    as_dtype(dtype)
  }
  nvl_fill(value, shape, dtype)
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


#' @title Broadcast Scalars to Common Shape
#' @description
#' Broadcast scalar tensors to match the shape of non-scalar tensors.
#' All non-scalar tensors must have the same shape.
#' @param ... ([`nv_tensor`])\cr
#'   Tensors to broadcast. Scalars will be broadcast to the common non-scalar shape.
#' @return (`list()` of [`nv_tensor`])
#' @export
nv_broadcast_scalars <- function(...) {
  args <- list(...)
  shapes <- lapply(args, \(x) shape(to_abstract(x)))
  non_scalar_shapes <- Filter(\(s) length(s) > 0L, shapes)

  if (length(non_scalar_shapes) == 0L) {
    return(args)
  }

  target_shape <- non_scalar_shapes[[1L]]
  if (!all(vapply(non_scalar_shapes, identical, logical(1L), target_shape))) {
    shapes <- paste0(sapply(shapes, shape2string), sep = ", ")
    cli_abort(
      "All non-scalar tensors must have the same shape, but got {shapes}. Use {.fn nv_broadcast_tensors} for general broadcasting." # nolint
    )
  }

  lapply(args, \(x) {
    if (length(shape(to_abstract(x))) == 0L) {
      nv_broadcast_to(x, target_shape)
    } else {
      x
    }
  })
}

#' @title Promote Tensors to a Common Dtype
#' @description
#' Promote tensors to a common data type, see [`common_dtype`] for more details.
#' @param ... ([`nv_tensor`])\cr
#'   Tensors to promote.
#' @return (`list()` of [`nv_tensor`])
#' @export
nv_promote_to_common <- function(...) {
  args <- list(...)
  avals <- lapply(args, to_abstract)
  tmp <- do.call(common_type_info, avals)
  cdt <- tmp[[1L]]
  ambiguous <- tmp[[2L]]
  out <- lapply(seq_along(args), \(i) {
    if (cdt == dtype(avals[[i]])) {
      args[[i]]
    } else {
      nvl_convert(args[[i]], dtype = cdt, ambiguous = ambiguous)
    }
  })
  return(out)
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
  shape <- Reduce(broadcast_shapes, lapply(args, \(x) shape(to_abstract(x))))
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
  shape_op <- shape(to_abstract(operand))
  if (!identical(shape_op, shape)) {
    broadcast_dimensions <- make_broadcast_dimensions(shape_op, shape)
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
  nvl_convert(operand, dtype = as_dtype(dtype), ambiguous = FALSE)
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
#'
#' @examples
#' # Comparison operators such `nv_eq`, `nv_le`, `nv_gt`, etc
#' # are nondifferentiable and contribute zero to gradients.
#' relu <- function(x) {
#'   nv_convert(x > nv_scalar(0), "f32")*x
#' }
#' # df/dx = 1 if x > 0 else 0
#' g_relu <- jit(gradient(relu, "x"))
#'
#' g_relu(nv_scalar(1, dtype = "f32"))
#' g_relu(nv_scalar(-1, dtype = "f32"))
#' @description
#' Binary operations on tensors.
#' @param lhs ([`nv_tensor`])
#' @param rhs ([`nv_tensor`])
#' @return [`nv_tensor`]
NULL


make_do_binary <- function(f) {
  function(lhs, rhs) {
    args <- nv_promote_to_common(lhs, rhs)
    args <- nv_broadcast_scalars(args[[1L]], args[[2L]])
    do.call(f, args)
  }
}

#' @rdname nv_binary_ops
#' @export
nv_add <- make_do_binary(nvl_add)

#' @rdname nv_binary_ops
#' @export
nv_mul <- make_do_binary(nvl_mul)

#' @rdname nv_binary_ops
#' @export
nv_sub <- make_do_binary(nvl_sub)

#' @rdname nv_binary_ops
#' @export
nv_div <- make_do_binary(nvl_div)

#' @rdname nv_binary_ops
#' @export
nv_pow <- make_do_binary(nvl_pow)

#' @rdname nv_binary_ops
#' @export
nv_eq <- make_do_binary(nvl_eq)

#' @rdname nv_binary_ops
#' @export
nv_ne <- make_do_binary(nvl_ne)

#' @rdname nv_binary_ops
#' @export
nv_gt <- make_do_binary(nvl_gt)

#' @rdname nv_binary_ops
#' @export
nv_ge <- make_do_binary(nvl_ge)

#' @rdname nv_binary_ops
#' @export
nv_lt <- make_do_binary(nvl_lt)

#' @rdname nv_binary_ops
#' @export
nv_le <- make_do_binary(nvl_le)

## Additional binary ops -------------------------------------------------------

#' @rdname nv_binary_ops
#' @export
nv_max <- make_do_binary(nvl_max)

#' @rdname nv_binary_ops
#' @export
nv_min <- make_do_binary(nvl_min)

#' @rdname nv_binary_ops
#' @export
nv_remainder <- make_do_binary(nvl_remainder)

#' @rdname nv_binary_ops
#' @export
nv_and <- make_do_binary(nvl_and)

#' @rdname nv_binary_ops
#' @export
nv_or <- make_do_binary(nvl_or)

#' @rdname nv_binary_ops
#' @export
nv_xor <- make_do_binary(nvl_xor)

#' @rdname nv_binary_ops
#' @export
nv_shift_left <- make_do_binary(nvl_shift_left)

#' @rdname nv_binary_ops
#' @export
nv_shift_right_logical <- make_do_binary(nvl_shift_right_logical)

#' @rdname nv_binary_ops
#' @export
nv_shift_right_arithmetic <- make_do_binary(nvl_shift_right_arithmetic)

#' @rdname nv_binary_ops
#' @export
nv_atan2 <- make_do_binary(nvl_atan2)


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
  args <- nv_promote_to_common(lhs, rhs)
  lhs <- args[[1L]]
  rhs <- args[[2L]]
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
  # from two random uniform variables u1 and u2 we can produce to normals z1, z2
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
  N <- nv_reshape(N, shape = shape_out)

  # return state and Normals N
  list(Theta[[1]], N)
}

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
