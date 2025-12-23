# This is the user-facing API containing the exported tensor operations.
#' @include primitives.R

# Special tensor creators

#' @title Iota
#' @description
#' Create a tensor with increasing values along a dimension.
#' @param dim (`integer(1)`)\cr
#'   The dimension along which to generate increasing values.
#' @template param_shape
#' @template param_dtype
#' @param start (`numeric(1)`)\cr
#'   The value to start the sequence at (default 1).
#' @export
nv_iota <- function(dim, shape, dtype = "i32", start = 1) {
  shape <- assert_shapevec(shape)
  nvl_iota(dim = dim, shape = shape, dtype = dtype, start = start)
}


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
#' @section Broadcasting:
#' All dimensions but the last two are broadcasted.
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
  shape_leading <- broadcast_shapes(head(shape_abstract(lhs), -2L), head(shape_abstract(rhs), -2L))

  shape_lhs <- c(shape_leading, tail(shape_abstract(lhs), 2L))
  shape_rhs <- c(shape_leading, tail(shape_abstract(rhs), 2L))

  if (!identical(shape_lhs, shape_abstract(lhs))) {
    lhs <- nv_broadcast_to(lhs, shape_lhs)
  }
  if (!identical(shape_rhs, shape_abstract(rhs))) {
    rhs <- nv_broadcast_to(rhs, shape_rhs)
  }

  nvl_dot_general(
    lhs,
    rhs,
    contracting_dims = list(ndims_abstract(lhs), ndims_abstract(rhs) - 1L),
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
