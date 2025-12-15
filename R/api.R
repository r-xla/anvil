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
nv_full <- function(value, shape, dtype = NULL) {
  dtype <- dtype %??%
    if (is.double(value)) {
      "f32"
    } else if (is.integer(value)) {
      "i32"
    } else if (is.logical(value)) {
      "pred"
    }
  nvl_full(value, shape, dtype)
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
  shapes <- lapply(args, \(x) shape(st(x)))
  non_scalar_shapes <- Filter(\(s) length(s) > 0L, shapes)

  if (length(non_scalar_shapes) == 0L) {
    return(args)
  }

  target_shape <- non_scalar_shapes[[1L]]
  if (!all(vapply(non_scalar_shapes, identical, logical(1L), target_shape))) {
    cli_abort(
      "All non-scalar tensors must have the same shape. Use {.fn nv_broadcast_tensors} for general broadcasting." # nolint
    )
  }

  lapply(args, \(x) {
    if (length(shape(st(x))) == 0L) {
      nv_broadcast_to(x, target_shape)
    } else {
      x
    }
  })
}

#' @title Promote Tensors to a Common Dtype
#' @description
#' Promote tensors to a common type.
#' @param ... ([`nv_tensor`])\cr
#'   Tensors to promote.
#' @return (`list()` of [`nv_tensor`])
#' @export
nv_promote_to_common <- function(...) {
  args <- list(...)
  avals <- lapply(args, st)
  tmp <- do.call(common_type_info, avals)
  cdt <- tmp[[1L]]
  out <- lapply(seq_along(args), \(i) {
    if (cdt == dtype(avals[[i]])) {
      args[[i]]
    } else {
      # don't promote ambiguity for now
      nvl_convert(args[[i]], dtype = cdt, ambiguous = FALSE)
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
  shape <- Reduce(broadcast_shapes, lapply(args, \(x) shape(st(x))))
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
  shape_op <- shape(st(operand))
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
  nvl_convert(operand, dtype = dtype, ambiguous = FALSE)
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


do_binary <- function(f, lhs, rhs) {
  args <- nv_promote_to_common(lhs, rhs)
  args <- nv_broadcast_scalars(args[[1L]], args[[2L]])
  do.call(f, args)
}

#' @rdname nv_binary_ops
#' @export
nv_add <- function(lhs, rhs) {
  do_binary(nvl_add, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_mul <- function(lhs, rhs) {
  do_binary(nvl_mul, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_sub <- function(lhs, rhs) {
  do_binary(nvl_sub, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_div <- function(lhs, rhs) {
  do_binary(nvl_div, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_pow <- function(lhs, rhs) {
  do_binary(nvl_pow, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_eq <- function(lhs, rhs) {
  do_binary(nvl_eq, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_ne <- function(lhs, rhs) {
  do_binary(nvl_ne, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_gt <- function(lhs, rhs) {
  do_binary(nvl_gt, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_ge <- function(lhs, rhs) {
  do_binary(nvl_ge, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_lt <- function(lhs, rhs) {
  do_binary(nvl_lt, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_le <- function(lhs, rhs) {
  do_binary(nvl_le, lhs, rhs)
}

## Additional binary ops -------------------------------------------------------

#' @rdname nv_binary_ops
#' @export
nv_max <- function(lhs, rhs) {
  do_binary(nvl_max, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_min <- function(lhs, rhs) {
  do_binary(nvl_min, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_remainder <- function(lhs, rhs) {
  do_binary(nvl_remainder, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_and <- function(lhs, rhs) {
  do_binary(nvl_and, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_or <- function(lhs, rhs) {
  do_binary(nvl_or, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_xor <- function(lhs, rhs) {
  do_binary(nvl_xor, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_shift_left <- function(lhs, rhs) {
  do_binary(nvl_shift_left, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_shift_right_logical <- function(lhs, rhs) {
  do_binary(nvl_shift_right_logical, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_shift_right_arithmetic <- function(lhs, rhs) {
  do_binary(nvl_shift_right_arithmetic, lhs, rhs)
}

#' @rdname nv_binary_ops
#' @export
nv_atan2 <- function(lhs, rhs) {
  do_binary(nvl_atan2, lhs, rhs)
}

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
