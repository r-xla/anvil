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
#' @param ... ([`tensorish`][tensorish])\cr
#'   Tensors to broadcast. Scalars will be broadcast to the common non-scalar shape.
#' @return (`list()` of [`tensorish`])\cr
#'   List of broadcasted tensors.
#' @export
nv_broadcast_scalars <- function(...) {
  args <- list(...)
  shapes <- lapply(args, shape_abstract)
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
    if (length(shape_abstract(x)) == 0L) {
      nv_broadcast_to(x, target_shape)
    } else {
      x
    }
  })
}

#' @title Promote Tensors to a Common Dtype
#' @description
#' Promote tensors to a common data type, see [`common_dtype`] for more details.
#' @param ... ([`tensorish`])\cr
#'   Tensors to promote.
#' @return (`list()` of [`tensorish`])
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
#' @param ... ([`tensorish`])\cr
#'   Tensors to broadcast.
#' @return (`list()` of [`tensorish`])
#' @export
nv_broadcast_tensors <- function(...) {
  args <- list(...)
  shape <- Reduce(broadcast_shapes, lapply(args, shape_abstract))
  lapply(args, nv_broadcast_to, shape = shape)
}

#' @title Broadcast
#' @description
#' Broadcast a tensor to a given shape using NumPy broadcasting rules.
#' @template param_operand
#' @param shape (`integer()`)\cr
#'   Output shape.
#' @return ([`tensorish`])
#' @export
nv_broadcast_to <- function(operand, shape) {
  shape_op <- shape_abstract(operand)
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
#' @return [`tensorish`]
#' @export
nv_convert <- function(operand, dtype) {
  nvl_convert(operand, dtype = as_dtype(dtype), ambiguous = FALSE)
}

#' @rdname nv_transpose
#' @export
nv_transpose <- function(x, permutation = NULL) {
  permutation <- permutation %??% rev(seq_len(ndims_abstract(x)))
  nvl_transpose(x, permutation)
}


#' @title Reshape
#' @description
#' Reshape a tensor.
#' Note that row-major order is used, which differs from R's column-major order.
#' @template param_operand
#' @param shape (`integer()`)\cr
#'   The new shape.
#' @return [`tensorish`]
#' @export
nv_reshape <- nvl_reshape

#' @title Concatenate
#' @description
#' Concatenate a variadic number of tensors.
#' @param ... tensors
#' @param dimension (`integer()`)\cr
#'   The dimension to concatenate along to. Other dimensions must be the same.
#' @return [`tensorish`]
#' @export
nv_concatenate <- nvl_concatenate

#' @title Slice
#' @description
#' return slice of operand.
#' @template param_operand
#' @param start_indices start of slice
#' @param limit_indices end of slice
#' @param strides stride size
#' @return [`tensorish`]
#' @export
nv_slice <- nvl_slice

#' @title Print Tensor
#' @description
#' Prints a tensor during JIT execution.
#' @template param_operand
#' @export
nv_print <- nvl_print

#' @title Select
#' @description
#' return values from true_value and false_value conditioned on pred
#' @param pred condition
#' @param true_value on true
#' @param false_value on false
#' @return [`tensorish`]
#' @export
nv_select <- nvl_select

## Binary ops ------------------------------------------------------------------

make_do_binary <- function(f) {
  function(lhs, rhs) {
    args <- nv_promote_to_common(lhs, rhs)
    args <- nv_broadcast_scalars(args[[1L]], args[[2L]])
    do.call(f, args)
  }
}

#' @title Addition
#' @description Element-wise addition of two tensors.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_add <- make_do_binary(nvl_add)

#' @title Multiplication
#' @description Element-wise multiplication of two tensors.
#' @param lhs ([`tensorish`])
#' @param rhs ([`tensorish`])
#' @return [`tensorish`]
#' @export
nv_mul <- make_do_binary(nvl_mul)

#' @title Subtraction
#' @description Element-wise subtraction of two tensors.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_sub <- make_do_binary(nvl_sub)

#' @title Division
#' @description Element-wise division of two tensors.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_div <- make_do_binary(nvl_div)

#' @title Power
#' @description Element-wise exponentiation of two tensors.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_pow <- make_do_binary(nvl_pow)

#' @title Equal
#' @description Element-wise equality comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_eq <- make_do_binary(nvl_eq)

#' @title Not Equal
#' @description Element-wise inequality comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_ne <- make_do_binary(nvl_ne)

#' @title Greater Than
#' @description Element-wise greater than comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_gt <- make_do_binary(nvl_gt)

#' @title Greater Than or Equal
#' @description Element-wise greater than or equal comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_ge <- make_do_binary(nvl_ge)

#' @title Less Than
#' @description Element-wise less than comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_lt <- make_do_binary(nvl_lt)

#' @title Less Than or Equal
#' @description Element-wise less than or equal comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_le <- make_do_binary(nvl_le)

#' @title Maximum
#' @description Element-wise maximum of two tensors.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_max <- make_do_binary(nvl_max)

#' @title Minimum
#' @description Element-wise minimum of two tensors.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_min <- make_do_binary(nvl_min)

#' @title Remainder
#' @description Element-wise remainder of division.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_remainder <- make_do_binary(nvl_remainder)

#' @title Logical And
#' @description Element-wise logical AND operation.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_and <- make_do_binary(nvl_and)

#' @title Logical Or
#' @description Element-wise logical OR operation.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_or <- make_do_binary(nvl_or)

#' @title Logical Xor
#' @description Element-wise logical XOR operation.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_xor <- make_do_binary(nvl_xor)

#' @title Shift Left
#' @description Element-wise bitwise left shift.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_shift_left <- make_do_binary(nvl_shift_left)

#' @title Logical Shift Right
#' @description Element-wise bitwise logical right shift.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_shift_right_logical <- make_do_binary(nvl_shift_right_logical)

#' @title Arithmetic Shift Right
#' @description Element-wise bitwise arithmetic right shift.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_shift_right_arithmetic <- make_do_binary(nvl_shift_right_arithmetic)

#' @title Arctangent 2
#' @description Element-wise two-argument arctangent.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nv_atan2 <- make_do_binary(nvl_atan2)


#' @title Bitcast Conversion
#' @name nv_bitcast_convert
#' @description
#' Reinterpret Bits
#' @template param_operand
#' @param dtype requested dtype
#' @export
nv_bitcast_convert <- nvl_bitcast_convert

## Unary ops ------------------------------------------------------------------

#' @title Negation
#' @description Element-wise negation.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_neg <- nvl_neg

#' @title Logical Not
#' @description Element-wise logical NOT operation.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_not <- nvl_not

#' @title Absolute Value
#' @description Element-wise absolute value.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_abs <- nvl_abs

#' @title Square Root
#' @description Element-wise square root.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_sqrt <- nvl_sqrt

#' @title Reciprocal Square Root
#' @description Element-wise reciprocal square root.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_rsqrt <- nvl_rsqrt

#' @title Natural Logarithm
#' @description Element-wise natural logarithm.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_log <- nvl_log

#' @title Hyperbolic Tangent
#' @description Element-wise hyperbolic tangent.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_tanh <- nvl_tanh

#' @title Tangent
#' @description Element-wise tangent.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_tan <- nvl_tan

#' @title Sine
#' @description Element-wise sine.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_sine <- nvl_sine

#' @title Cosine
#' @description Element-wise cosine.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_cosine <- nvl_cosine

#' @title Floor
#' @description Element-wise floor (round toward negative infinity).
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_floor <- nvl_floor

#' @title Ceiling
#' @description Element-wise ceiling (round toward positive infinity).
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_ceil <- nvl_ceil

#' @title Sign
#' @description Element-wise sign function.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_sign <- nvl_sign

#' @title Exponential
#' @description Element-wise exponential function.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nv_exp <- nvl_exp

#' @title Round
#' @description Element-wise rounding.
#' @template param_operand
#' @param method (`character(1)`)\cr
#'   Method to use for rounding.
#'   Either `"nearest_even"` (default) or `"afz"` (away from zero).
#' @return [`tensorish`]
#' @export
nv_round <- nvl_round

## Other operations -----------------------------------------------------------

#' @title Matrix Multiplication
#' @description
#' Matrix multiplication of two tensors.
#' @section Shapes:
#' - `lhs`: `(b1, ..., bk, m, n)`
#' - `rhs`: `(b1, ..., bk, n, p)`
#' - output: `(b1, ..., bk, m, p)`
#' @param lhs ([`tensorish`])
#' @param rhs ([`tensorish`])
#' @return [`tensorish`]
#' @export
nv_matmul <- function(lhs, rhs) {
  args <- nv_promote_to_common(lhs, rhs)
  lhs <- args[[1L]]
  rhs <- args[[2L]]
  if (ndims_abstract(lhs) < 2L) {
    cli_abort("lhs of matmul must have at least 2 dimensions")
  }
  if (ndims_abstract(rhs) < 2L) {
    cli_abort("rhs of matmul must have at least 2 dimensions")
  }
  nbatch <- ndims_abstract(lhs) - 2L
  nvl_dot_general(
    lhs,
    rhs,
    contracting_dims = list(ndims_abstract(lhs), ndims_abstract(rhs) - 1L),
    batching_dims = list(seq_len(nbatch), seq_len(nbatch))
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
#' @return [`tensorish`]
#' @export
nv_reduce_sum <- nvl_reduce_sum

#' @rdname nv_reduce_ops
#' @export
nv_reduce_mean <- function(operand, dims, drop = TRUE) {
  # TODO: division by zero?
  nelts <- prod(shape_abstract(operand)[dims])
  # TODO: Should just be able to do use autocasting and divide by nelts scalar
  nv_reduce_sum(operand, dims, drop) / nelts
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
#' @description
#' generate random bits of desired shape and dtype
#' @param initial_state state seed
#' @param rng_algorithm one of 'DEFAULT', 'THREE_FRY', 'PHILOX'
#' @param dtype datatype of output
#' @param shape_out output shape
#' @export
nv_rng_bit_generator <- nvl_rng_bit_generator

#' @title Random Uniform Numbers
#' @description
#' generate random uniform numbers in ]lower, upper[
#' @param initial_state ([`tensorish`])\cr
#'   Tensor of type `ui64[2]`.
#' @template param_dtype
#' @param dtype output dtype either "f32" or "f64"
#' @param shape output shape
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
  checkmate::assertIntegerish(shape, lower = 1, min.len = 1, any.missing = FALSE)

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
  checkmate::assertIntegerish(shape, lower = 1, min.len = 1, any.missing = FALSE)
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

# Higher order primitives

#' @title If
#' @description
#' Functional if statement.
#' @param pred ([`tensorish`])\cr
#'   Flag.
#' @param true (NSE)\cr
#'   Expression to evaluate if the condition is true.
#' @param false (NSE)\cr
#'   Expression to evaluate if the condition is false.
#' @return [`tensorish`]
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
#' @return [`tensorish`]
#' @export
nv_while <- nvl_while
