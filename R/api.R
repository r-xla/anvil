# This is the user-facing API containing the exported array operations.
#' @include primitives.R

# Special array creators

#' @title Fill Constant
#' @description
#' Creates an array filled with a scalar value. More memory-efficient than
#' `nv_array(value, shape = shape)` for large arrays.
#'
#' `nv_fill_like()` is a variant where `dtype`, `shape`, `ambiguous`, and
#' `device` default to those of `like`.
#' @param value (`numeric(1)`)\cr
#'   Scalar value to fill the array with.
#' @param shape (`integer()`)\cr
#'   Shape of the output array.
#' @param dtype (`character(1)` | `NULL`)\cr
#'   Data type.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_fill_like()`).
#' @template param_ambiguous
#' @template param_device
#' @return [`arrayish`]\cr
#'   Has the given `shape` and `dtype`.
#' @seealso [prim_fill()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_fill(0, shape = c(2, 3))
#' x <- nv_array(matrix(1:6, nrow = 2))
#' nv_fill_like(x, 0)
#' @export
nv_fill <- function(value, shape, dtype = NULL, ambiguous = FALSE, device = NULL) {
  if (!is_valid_r_lit(value)) {
    cli_abort(
      "{.arg value} must be an R vector of length 1 of type double, integer, or logical, not {.cls {class(value)[1]}}."
    )
  }
  dtype <- if (is.null(dtype)) {
    default_dtype(value)
  } else {
    as_dtype(dtype)
  }
  prim_fill(value, shape, dtype, ambiguous, device = device)
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
#' Broadcast scalar arrays to match the shape of non-scalar arrays.
#' All non-scalar arrays must have the same shape.
#' @param ... ([`arrayish`][arrayish])\cr
#'   Arrays to broadcast. Scalars will be broadcast to the common non-scalar shape.
#' @return (`list()` of [`arrayish`])\cr
#'   List of broadcasted arrays.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' # scalar 1 is broadcast to shape [3]
#' nv_broadcast_scalars(x, nv_scalar(1))
#' @export
nv_broadcast_scalars <- function(...) {
  args <- as_anvl_arrays(...)
  shapes <- lapply(args, shape)
  non_scalar_shapes <- Filter(\(s) length(s) > 0L, shapes)

  if (length(non_scalar_shapes) == 0L) {
    return(args)
  }

  target_shape <- non_scalar_shapes[[1L]]
  if (!all(vapply(non_scalar_shapes, identical, logical(1L), target_shape))) {
    shapes <- paste0(sapply(shapes, shape2string), sep = ", ")
    cli_abort(
      "All non-scalar arrays must have the same shape, but got {shapes}. Use {.fn nv_broadcast_arrays} for general broadcasting." # nolint
    )
  }

  lapply(args, \(x) {
    if (length(shape(x)) == 0L) {
      nv_broadcast_to(x, target_shape)
    } else {
      x
    }
  })
}

#' @title Promote Arrays to a Common Dtype
#' @description
#' Promote arrays to a common data type, see [`common_dtype`] for more details.
#' @param ... ([`arrayish`])\cr
#'   Arrays to promote.
#' @return (`list()` of [`arrayish`])
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1L)
#' y <- nv_array(1.5)
#' # integer is promoted to float
#' nv_promote_to_common(x, y)
#' @export
nv_promote_to_common <- function(...) {
  args <- as_anvl_arrays(...)
  tmp <- do.call(common_type_info, args)
  cdt <- tmp[[1L]]
  ambiguous <- tmp[[2L]]
  out <- lapply(seq_along(args), \(i) {
    if (cdt == dtype(args[[i]])) {
      args[[i]]
    } else {
      prim_convert(args[[i]], dtype = cdt, ambiguous = ambiguous)
    }
  })
  return(out)
}

#' @title Broadcast Arrays to a Common Shape
#' @description
#' Broadcasts arrays to a common shape using NumPy-style broadcasting rules.
#'
#' @section Broadcasting Rules:
#' 1. If the arrays have different numbers of dimensions, prepend size-1
#'    dimensions to the shorter shape.
#' 2. For each dimension: if the sizes match, keep them; if one is 1, expand
#'    it to the other's size; otherwise raise an error.
#'
#' @param ... ([`arrayish`])\cr
#'   Arrays to broadcast.
#' @return (`list()` of [`arrayish`])\cr
#'   List of arrays, all with the same shape.
#' @seealso [nv_broadcast_scalars()], [nv_broadcast_to()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(1:6, nrow = 2))
#' y <- nv_array(c(10, 20, 30))
#' nv_broadcast_arrays(x, y)
#' @export
nv_broadcast_arrays <- function(...) {
  args <- as_anvl_arrays(...)
  shape <- Reduce(broadcast_shapes, lapply(args, shape))
  lapply(args, nv_broadcast_to, shape = shape)
}

#' @title Broadcast to Shape
#' @description
#' Broadcasts an array to a target shape using NumPy-style broadcasting rules.
#' @template param_operand
#' @param shape (`integer()`)\cr
#'   Target shape. Each existing dimension must either match or be 1.
#' @return [`arrayish`]\cr
#'   Has the given `shape` and the same data type as `operand`.
#' @seealso [nv_broadcast_arrays()], [nv_broadcast_scalars()],
#'   [prim_broadcast_in_dim()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' nv_broadcast_to(x, shape = c(2, 3))
#' @export
nv_broadcast_to <- function(operand, shape) {
  operand <- as_anvl_array(operand)
  shape_op <- shape(operand)
  if (!identical(shape_op, shape)) {
    broadcast_dimensions <- make_broadcast_dimensions(shape_op, shape)
    prim_broadcast_in_dim(operand, shape, broadcast_dimensions)
  } else {
    operand
  }
}

#' @title Convert Data Type
#' @description
#' Converts the elements of an array to a different data type.
#' Returns the input unchanged if it already has the target type.
#' @template param_operand
#' @template param_dtype
#' @return [`arrayish`]\cr
#'   Has the given `dtype` and the same shape as `operand`.
#' @seealso [prim_convert()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1L, 2L, 3L))
#' nv_convert(x, dtype = "f32")
#' @export
nv_convert <- function(operand, dtype) {
  operand <- as_anvl_array(operand)
  if (dtype(operand) != as_dtype(dtype)) {
    prim_convert(operand, dtype = as_dtype(dtype), ambiguous = FALSE)
  } else {
    operand
  }
}

#' @rdname nv_transpose
#' @export
nv_transpose <- function(x, permutation = NULL) {
  x <- as_anvl_array(x)
  permutation <- permutation %||% rev(seq_len(ndims(x)))
  prim_transpose(x, permutation)
}


#' @title Reshape
#' @description
#' Reshapes an array to a new shape without changing the underlying data.
#' Returns the input unchanged if it already has the target shape.
#' @details
#' Note that row-major order is used, which differs from R's column-major order.
#' @template param_operand
#' @param shape (`integer()`)\cr
#'   Target shape. Must have the same number of elements as `operand`.
#' @return [`arrayish`]\cr
#'   Has the given `shape` and the same data type as `operand`.
#' @seealso [prim_reshape()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:6)
#' nv_reshape(x, c(2, 3))
#' @export
nv_reshape <- function(operand, shape) {
  operand <- as_anvl_array(operand)
  if (!identical(shape(operand), shape)) {
    prim_reshape(operand, shape)
  } else {
    operand
  }
}

#' @title Concatenate
#' @description
#' Concatenates arrays along a dimension. Operands are promoted to a common
#' data type and scalars are broadcast before concatenation.
#' @param ... ([`arrayish`])\cr
#'   Arrays to concatenate. Must have the same shape except along `dimension`.
#' @param dimension (`integer(1)` | `NULL`)\cr
#'   Dimension along which to concatenate.
#'   If `NULL` (default), assumes all inputs are at most 1-D and concatenates along dimension 1.
#' @return [`arrayish`]\cr
#'   Has the common data type and a shape matching the inputs in all
#'   dimensions except `dimension`, which is the sum of input sizes.
#' @seealso [prim_concatenate()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' nv_concatenate(x, y)
#' @export
nv_concatenate <- function(..., dimension = NULL) {
  args <- do.call(nv_promote_to_common, list(...))
  shapes <- lapply(args, shape)
  ranks <- lengths(shapes)
  non_scalar_shapes <- shapes[ranks > 0L]
  n_scalars <- sum(ranks == 0L)
  assert_int(dimension, lower = 1L, upper = max(max(ranks), 1L), null.ok = max(ranks) <= 1L)
  dimension <- dimension %||% 1L

  non_scalar_shapes_without_dim <- lapply(non_scalar_shapes, \(shape) {
    shape[-dimension]
  })
  if (length(non_scalar_shapes) && length(unique(non_scalar_shapes_without_dim)) != 1L) {
    cli_abort(c(
      "All non-scalar arrays must have the same shape (except for the concatenation dimension)",
      x = "Got shapes {shapes2string(shapes)} and dimension {dimension}"
    ))
  }
  size_out_dimension <- n_scalars + sum(vapply(non_scalar_shapes, \(shape) shape[dimension], integer(1L)))

  out_shape <- if (length(non_scalar_shapes)) {
    x <- non_scalar_shapes[[1L]]
    x[dimension] <- size_out_dimension
  } else {
    n_scalars
  }
  out_shape_dim_is_one <- out_shape
  out_shape_dim_is_one[dimension] <- 1L
  args <- lapply(args, \(arg) {
    if (ndims(arg) == 0L) {
      nv_broadcast_to(arg, out_shape_dim_is_one)
    } else {
      arg
    }
  })
  rlang::exec(prim_concatenate, !!!args, dimension = dimension)
}

#' @title Static Slice
#' @description
#' Extracts a slice from an array using static (compile-time) indices.
#' For dynamic indexing, use [nv_subset()] instead.
#' @template param_operand
#' @param start_indices (`integer()`)\cr
#'   Start indices (inclusive), one per dimension.
#' @param limit_indices (`integer()`)\cr
#'   End indices (inclusive), one per dimension.
#' @param strides (`integer()`)\cr
#'   Step sizes, one per dimension. A stride of 1 selects every element.
#' @return [`arrayish`]\cr
#'   Has the same data type as `operand`.
#' @seealso [nv_subset()], [prim_static_slice()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:10)
#' nv_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 1L)
#' @export
nv_static_slice <- prim_static_slice

#' @title Print Array
#' @description
#' Prints an array value to the console during JIT execution and returns the
#' input unchanged. Useful for debugging.
#' @template param_operand
#' @return [`arrayish`]\cr
#'   Returns `operand` unchanged.
#' @seealso [prim_print()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' nv_print(x)
#' @export
nv_print <- prim_print

#' @title Conditional Element Selection
#' @description
#' Selects elements from `true_value` or `false_value` based on `pred`,
#' analogous to R's [ifelse()].
#' @param pred ([`arrayish`] of boolean type)\cr
#'   Predicate array. Must be scalar or have the same shape as the
#'   non-scalar arguments.
#' @param true_value,false_value ([`arrayish`])\cr
#'   Values to return where `pred` is `TRUE` / `FALSE`.
#'   `true_value` and `false_value` are
#'   [promoted to a common data type][nv_promote_to_common()].
#'   Scalars (including `pred`) are
#'   [broadcast][nv_broadcast_scalars()] to the shape of the non-scalar arguments.
#' @return [`arrayish`]\cr
#'   Has the common data type of `true_value` and `false_value` and the
#'   shape of the non-scalar arguments.
#' @seealso [prim_ifelse()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' pred <- nv_array(c(TRUE, FALSE, TRUE))
#' nv_ifelse(pred, nv_array(c(1, 2, 3)), nv_array(c(4, 5, 6)))
#' # scalar branches are broadcast and promoted to a common dtype
#' nv_ifelse(pred, nv_scalar(1L), nv_scalar(0.5))
#' @export
nv_ifelse <- function(pred, true_value, false_value) {
  promoted <- nv_promote_to_common(true_value, false_value)
  args <- nv_broadcast_scalars(pred, promoted[[1L]], promoted[[2L]])
  prim_ifelse(args[[1L]], args[[2L]], args[[3L]])
}

## Binary ops ------------------------------------------------------------------

make_do_binary <- function(f) {
  function(lhs, rhs) {
    args <- nv_promote_to_common(lhs, rhs)
    args <- nv_broadcast_scalars(args[[1L]], args[[2L]])
    do.call(f, args)
  }
}

#' @title Addition
#' @description
#' Adds two arrays element-wise. You can also use the `+` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_add()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' x + y
#' @export
nv_add <- make_do_binary(prim_add)

#' @title Multiplication
#' @description
#' Multiplies two arrays element-wise. You can also use the `*` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_mul()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' x * y
#' @export
nv_mul <- make_do_binary(prim_mul)

#' @title Subtraction
#' @description
#' Subtracts two arrays element-wise. You can also use the `-` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_sub()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(4, 5, 6))
#' y <- nv_array(c(1, 2, 3))
#' x - y
#' @export
nv_sub <- make_do_binary(prim_sub)

#' @title Division
#' @description
#' Divides two arrays element-wise. You can also use the `/` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_div()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(10, 20, 30))
#' y <- nv_array(c(2, 5, 10))
#' x / y
#' @export
nv_div <- make_do_binary(prim_div)

#' @title Power
#' @description
#' Raises `lhs` to the power of `rhs` element-wise. You can also use the `^` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_pow()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(2, 3, 4))
#' y <- nv_array(c(3, 2, 1))
#' x ^ y
#' @export
nv_pow <- make_do_binary(prim_pow)

#' @title Equal
#' @description
#' Element-wise equality comparison. You can also use the `==` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_eq()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(1, 3, 2))
#' x == y
#' @export
nv_eq <- make_do_binary(prim_eq)

#' @title Not Equal
#' @description
#' Element-wise inequality comparison. You can also use the `!=` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_ne()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(1, 3, 2))
#' x != y
#' @export
nv_ne <- make_do_binary(prim_ne)

#' @title Greater Than
#' @description
#' Element-wise greater than comparison. You can also use the `>` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_gt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' x > y
#' @export
nv_gt <- make_do_binary(prim_gt)

#' @title Greater Than or Equal
#' @description
#' Element-wise greater than or equal comparison. You can also use the `>=` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_ge()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' x >= y
#' @export
nv_ge <- make_do_binary(prim_ge)

#' @title Less Than
#' @description
#' Element-wise less than comparison. You can also use the `<` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_lt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' x < y
#' @export
nv_lt <- make_do_binary(prim_lt)

#' @title Less Than or Equal
#' @description
#' Element-wise less than or equal comparison. You can also use the `<=` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_le()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' x <= y
#' @export
nv_le <- make_do_binary(prim_le)

#' @title Maximum
#' @description
#' Element-wise maximum of two arrays.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_max()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 5, 3))
#' y <- nv_array(c(4, 2, 6))
#' nv_max(x, y)
#' @export
nv_max <- make_do_binary(prim_max)

#' @title Minimum
#' @description
#' Element-wise minimum of two arrays.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_min()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 5, 3))
#' y <- nv_array(c(4, 2, 6))
#' nv_min(x, y)
#' @export
nv_min <- make_do_binary(prim_min)

#' @title Remainder
#' @description
#' Element-wise remainder of division. You can also use the `%%` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_remainder()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(7, 8, 9))
#' y <- nv_array(c(3, 3, 4))
#' x %% y
#' @export
nv_remainder <- make_do_binary(prim_remainder)

#' @title Logical And
#' @description
#' Element-wise logical AND. You can also use the `&` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_and()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' x & y
#' @export
nv_and <- make_do_binary(prim_and)

#' @title Logical Or
#' @description
#' Element-wise logical OR. You can also use the `|` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_or()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' x | y
#' @export
nv_or <- make_do_binary(prim_or)

#' @title Logical Xor
#' @description
#' Element-wise logical XOR.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_xor()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' nv_xor(x, y)
#' @export
nv_xor <- make_do_binary(prim_xor)

#' @title Shift Left
#' @description
#' Element-wise left bit shift.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_shift_left()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1L, 2L, 4L))
#' y <- nv_array(c(1L, 2L, 1L))
#' nv_shift_left(x, y)
#' @export
nv_shift_left <- make_do_binary(prim_shift_left)

#' @title Logical Shift Right
#' @description
#' Element-wise logical right bit shift.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_shift_right_logical()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(8L, 16L, 32L))
#' y <- nv_array(c(1L, 2L, 3L))
#' nv_shift_right_logical(x, y)
#' @export
nv_shift_right_logical <- make_do_binary(prim_shift_right_logical)

#' @title Arithmetic Shift Right
#' @description
#' Element-wise arithmetic right bit shift.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_shift_right_arithmetic()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(8L, -16L, 32L))
#' y <- nv_array(c(1L, 2L, 3L))
#' nv_shift_right_arithmetic(x, y)
#' @export
nv_shift_right_arithmetic <- make_do_binary(prim_shift_right_arithmetic)

#' @title Arctangent 2
#' @description
#' Element-wise two-argument arctangent, i.e. the angle (in radians) between the positive
#' x-axis and the point `(rhs, lhs)`.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_atan2()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' y <- nv_array(c(1, 0, -1))
#' x <- nv_array(c(0, 1, 0))
#' nv_atan2(y, x)
#' @export
nv_atan2 <- make_do_binary(prim_atan2)


#' @title Bitcast Conversion
#' @name nv_bitcast_convert
#' @description
#' Reinterprets the bits of an array as a different data type without modifying
#' the underlying data. If the target type is narrower, an extra trailing
#' dimension is added; if wider, the last dimension is consumed.
#' @template param_operand
#' @param dtype (`character(1)` | [`DataType`])\cr
#'   Target data type.
#' @return [`arrayish`]\cr
#'   Has the given `dtype`.
#' @seealso [prim_bitcast_convert()] for the underlying primitive, [nv_convert()]
#'   for value-preserving type conversion.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1L)
#' prim_bitcast_convert(x, dtype = "i8")
#' @export
nv_bitcast_convert <- prim_bitcast_convert

## Unary ops ------------------------------------------------------------------

#' @title Negation
#' @description
#' Negates an array element-wise. You can also use the unary `-` operator.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_negate()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, -2, 3))
#' -x
#' @export
nv_negate <- prim_negate

#' @title Logical Not
#' @description
#' Element-wise logical NOT. You can also use the `!` operator.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_not()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' !x
#' @export
nv_not <- prim_not

#' @title Absolute Value
#' @description
#' Element-wise absolute value. You can also use `abs()`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_abs()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 2, -3))
#' abs(x)
#' @export
nv_abs <- prim_abs

#' @title Square Root
#' @description
#' Element-wise square root. You can also use `sqrt()`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_sqrt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 4, 9))
#' sqrt(x)
#' @export
nv_sqrt <- prim_sqrt

#' @title Reciprocal Square Root
#' @description
#' Element-wise reciprocal square root, i.e. `1 / sqrt(x)`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_rsqrt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 4, 9))
#' nv_rsqrt(x)
#' @export
nv_rsqrt <- prim_rsqrt

#' @title Natural Logarithm
#' @description
#' Element-wise natural logarithm. You can also use `log()`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_log()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2.718, 7.389))
#' log(x)
#' @export
nv_log <- prim_log

#' @title Hyperbolic Tangent
#' @description
#' Element-wise hyperbolic tangent. You can also use `tanh()`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_tanh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' tanh(x)
#' @export
nv_tanh <- prim_tanh

#' @title Tangent
#' @description
#' Element-wise tangent. You can also use `tan()`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_tan()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 0.5, 1))
#' tan(x)
#' @export
nv_tan <- prim_tan

#' @title Sine
#' @description
#' Element-wise sine. You can also use `sin()`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_sine()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, pi / 2, pi))
#' sin(x)
#' @export
nv_sine <- prim_sine

#' @title Cosine
#' @description
#' Element-wise cosine. You can also use `cos()`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_cosine()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, pi / 2, pi))
#' cos(x)
#' @export
nv_cosine <- prim_cosine

#' @title Floor
#' @description
#' Element-wise floor (round toward negative infinity). You can also use `floor()`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_floor()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1.2, 2.7, -1.5))
#' floor(x)
#' @export
nv_floor <- prim_floor

#' @title Ceiling
#' @description
#' Element-wise ceiling (round toward positive infinity). You can also use `ceiling()`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_ceil()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1.2, 2.7, -1.5))
#' ceiling(x)
#' @export
nv_ceil <- prim_ceil

#' @title Sign
#' @description
#' Element-wise sign function. You can also use `sign()`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_sign()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-3, 0, 5))
#' sign(x)
#' @export
nv_sign <- prim_sign

#' @title Exponential
#' @description
#' Element-wise exponential. You can also use `exp()`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_exp()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 1, 2))
#' exp(x)
#' @export
nv_exp <- prim_exp

#' @title Exponential Minus One
#' @description
#' Element-wise `exp(x) - 1`, more accurate for small `x`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_expm1()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 0.001, 1))
#' nv_expm1(x)
#' @export
nv_expm1 <- prim_expm1

#' @title Log Plus One
#' @description
#' Element-wise `log(1 + x)`, more accurate for small `x`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_log1p()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 0.001, 1))
#' nv_log1p(x)
#' @export
nv_log1p <- prim_log1p

#' @title Cube Root
#' @description
#' Element-wise cube root.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_cbrt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 8, 27))
#' nv_cbrt(x)
#' @export
nv_cbrt <- prim_cbrt

#' @title Logistic (Sigmoid)
#' @description
#' Element-wise logistic sigmoid: `1 / (1 + exp(-x))`.
#' @template param_operand
#' @template return_unary
#' @seealso [prim_logistic()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-2, 0, 2))
#' nv_logistic(x)
#' @export
nv_logistic <- prim_logistic

#' @title Is Finite
#' @description
#' Element-wise check if values are finite (not `Inf`, `-Inf`, or `NaN`).
#' @template param_operand
#' @template return_unary_boolean
#' @seealso [prim_is_finite()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, Inf, NaN, -Inf, 0))
#' nv_is_finite(x)
#' @export
nv_is_finite <- prim_is_finite

#' @title Population Count
#' @description
#' Element-wise population count (number of set bits).
#' @template param_operand
#' @template return_unary
#' @seealso [prim_popcnt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(7L, 3L, 15L))
#' nv_popcnt(x)
#' @export
nv_popcnt <- prim_popcnt

#' @title Clamp
#' @description
#' Element-wise clamp: `max(min_val, min(operand, max_val))`.
#' Converts `min_val` and `max_val` to the data type of `operand`.
#' @details
#' The underlying stableHLO function already broadcasts scalars, so no need to broadcast manually.
#' @param min_val,max_val ([`arrayish`])\cr
#'   Minimum and maximum values (scalar or same shape as `operand`).
#' @template param_operand
#' @template return_unary
#' @seealso [prim_clamp()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0.5, 2))
#' nv_clamp(nv_scalar(0), x, nv_scalar(1))
#' @export
nv_clamp <- function(min_val, operand, max_val) {
  args <- as_anvl_arrays(min_val, operand, max_val)
  min_val <- args[[1L]]
  operand <- args[[2L]]
  max_val <- args[[3L]]
  op_dtype <- dtype(operand)
  min_val <- nv_convert(min_val, op_dtype)
  max_val <- nv_convert(max_val, op_dtype)
  prim_clamp(min_val, operand, max_val)
}

#' @title Reverse
#' @description
#' Reverses the order of elements along specified dimensions.
#' @template param_operand
#' @param dims (`integer()`)\cr
#'   Dimensions to reverse.
#' @return [`arrayish`]\cr
#'   Has the same shape and data type as `operand`.
#' @seealso [prim_reverse()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3, 4, 5))
#' nv_reverse(x, dims = 1L)
#' @export
nv_reverse <- prim_reverse

#' @title Iota
#' @description
#' Creates an array with values increasing along the specified dimension,
#' starting from `start`.
#'
#' `nv_iota_like()` is a variant where `dtype`, `shape`, `ambiguous`, and
#' `device` default to those of `like`.
#' @param dim (`integer(1)`)\cr
#'   Dimension along which values increase.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_iota_like()`).
#' @template param_dtype
#' @template param_shape
#' @param start (`integer(1)`)\cr
#'   Starting value (default 1).
#' @template param_ambiguous
#' @template param_device
#' @return [`arrayish`]\cr
#'   Has the given `dtype` and `shape`.
#' @seealso [nv_seq()] for a simpler 1-D sequence, [prim_iota()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_iota(dim = 1L, dtype = "i32", shape = 5L)
#' x <- nv_array(matrix(0L, nrow = 2, ncol = 3))
#' nv_iota_like(x, dim = 1L)
#' @export
nv_iota <- prim_iota

#' @title Sequence
#' @description
#' Creates a 1-D array with values from `start` to `end` (inclusive).
#'
#' Without `steps`, behaves like R's `seq(start, end)` producing integer values.
#' With `steps`, produces `steps` evenly spaced values (like `seq(start, end, length.out = steps)`).
#'
#' `nv_seq_like()` is a variant where `dtype`, `ambiguous`, and `device`
#' default to those of `like`.
#' @param start,end (`numeric(1)`)\cr
#'   Start and end values. When `steps` is `NULL`, must satisfy `start <= end`.
#' @param steps (`integer(1)` or `NULL`)\cr
#'   Number of evenly spaced values to generate. Must be at least 1.
#'   When `NULL` (default), generates consecutive integer values from `start` to `end`.
#' @param dtype (`character(1)`)\cr
#'   Data type. Default `"i32"` when `steps` is `NULL`, `"f32"` when `steps` is given.
#'   For `nv_seq_like()`, `NULL` uses `dtype(like)`.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_seq_like()`).
#' @template param_ambiguous
#' @template param_device
#' @return [`arrayish`]\cr
#'   1-D array of length `end - start + 1`.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_seq(3, 7)
#' x <- nv_array(c(1, 2, 3), dtype = "f64")
#' nv_seq_like(x, 1, 5)
#' @export
nv_seq <- function(start, end, steps = NULL, dtype = NULL, ambiguous = FALSE, device = NULL) {
  if (is.null(steps)) {
    dtype <- dtype %||% "i32"
    assert_int(start)
    assert_int(end)
    assert(start <= end)
    return(nv_iota(
      shape = end - start + 1,
      dtype = dtype,
      ambiguous = ambiguous,
      dim = 1L,
      start = start,
      device = device
    ))
  }
  dtype <- dtype %||% "f32"
  assert_int(steps, lower = 1L)
  if (steps == 1L) {
    return(nv_fill(start, 1L, dtype = dtype, device = device))
  }
  indices <- nv_iota(dim = 1L, shape = steps, dtype = dtype, start = 0L, device = device)
  indices * ((end - start) / (steps - 1L)) + start
}

#' @title Pad
#' @description
#' Pads an array with a given value at the edges and optionally between elements.
#' @template param_operand
#' @param padding_value ([`arrayish`])\cr
#'   Scalar value to use for padding. Must have the same dtype as `operand`.
#' @param edge_padding_low (`integer()`)\cr
#'   Amount of padding to add at the start of each dimension.
#' @param edge_padding_high (`integer()`)\cr
#'   Amount of padding to add at the end of each dimension.
#' @param interior_padding (`integer()` | `NULL`)\cr
#'   Amount of padding to add between elements in each dimension.
#'   If `NULL` (default), no interior padding is applied.
#' @return [`arrayish`]\cr
#'   Has the same data type as `operand`.
#' @seealso [prim_pad()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' nv_pad(x, nv_scalar(0), edge_padding_low = 2L, edge_padding_high = 1L)
#' @export
nv_pad <- function(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding = NULL) {
  args <- as_anvl_arrays(operand, padding_value)
  operand <- args[[1L]]
  padding_value <- args[[2L]]
  rank <- ndims(operand)
  if (is.null(interior_padding)) {
    interior_padding <- rep(0L, rank)
  }
  prim_pad(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding)
}

#' @title Round
#' @description
#' Element-wise rounding. You can also use the `round()` generic.
#' @template param_operand
#' @param method (`character(1)`)\cr
#'   Rounding method.
#'   Either `"nearest_even"` (default) or `"afz"` (away from zero).
#' @template return_unary
#' @seealso [prim_round()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1.4, 2.5, 3.6))
#' round(x)
#' @export
nv_round <- prim_round

## Other operations -----------------------------------------------------------

#' @title Matrix Multiplication
#' @description
#' Matrix multiplication of two arrays. You can also use the `%*%` operator.
#' Supports batched matrix multiplication when inputs have more than 2 dimensions.
#' @section Shapes:
#' - `lhs`: `(b1, ..., bk, m, n)`
#' - `rhs`: `(b1, ..., bk, n, p)`
#' - output: `(b1, ..., bk, m, p)`
#' @param lhs,rhs ([`arrayish`])\cr
#'   Arrays with at least 2 dimensions.
#'   Operands are [promoted to a common data type][nv_promote_to_common()].
#' @return [`arrayish`]
#' @seealso [prim_dot_general()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(1:6, nrow = 2))
#' y <- nv_array(matrix(1:6, nrow = 3))
#' x %*% y
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
  nbatch <- ndims(lhs) - 2L
  prim_dot_general(
    lhs,
    rhs,
    contracting_dims = list(ndims(lhs), ndims(rhs) - 1L),
    batching_dims = list(seq_len(nbatch), seq_len(nbatch))
  )
}

#' @title Cholesky Decomposition
#' @description
#' Computes the Cholesky decomposition of a symmetric positive-definite matrix.
#' Supports batched inputs: dimensions before the last two are batch dimensions.
#' @param a ([`arrayish`])\cr
#'   Symmetric positive-definite matrix with at least 2 dimensions.
#'   The last two dimensions form the square matrix; any leading dimensions
#'   are batch dimensions.
#' @param lower (`logical(1)`)\cr
#'   If `TRUE` (default), compute the lower triangular factor `L` such that
#'   `a = L %*% t(L)`. If `FALSE`, compute the upper triangular factor `U`
#'   such that `a = t(U) %*% U`.
#' @return [`arrayish`]\cr
#'   Triangular matrix with the same shape and data type as the input.
#' @seealso [nv_solve()], [prim_cholesky()]
#' @examplesIf pjrt::plugins_downloaded()
#' a <- nv_array(matrix(c(4, 2, 2, 3), nrow = 2), dtype = "f32")
#' nv_cholesky(a)
#' @export
nv_cholesky <- function(a, lower = TRUE) {
  a <- as_anvl_array(a)
  prim_cholesky(a, lower = lower)
}

#' @title Solve Linear System
#' @description
#' Solves the linear system `a %*% x = b` for `x`, where `a` is a symmetric
#' positive-definite matrix. Uses Cholesky decomposition internally.
#' Supports batched inputs: `a` and `b` must have the same batch dimensions
#' (all dimensions before the last two).
#' @section Shapes:
#' - `a`: `(..., n, n)`
#' - `b`: `(..., n, k)`
#' - output: same shape as `b`
#'
#' where `...` are zero or more batch dimensions that must match between
#' `a` and `b`.
#' @param a ([`arrayish`])\cr
#'   Symmetric positive-definite matrix.
#' @param b ([`arrayish`])\cr
#'   Right-hand side matrix or vector. Must have the same data type and batch
#'   dimensions as `a`.
#' @return [`arrayish`]\cr
#'   The solution `x` such that `a %*% x = b`.
#' @seealso [nv_cholesky()], [prim_cholesky()], [prim_triangular_solve()]
#' @examplesIf pjrt::plugins_downloaded()
#' a <- nv_array(matrix(c(4, 2, 2, 3), nrow = 2), dtype = "f32")
#' b <- nv_array(matrix(c(1, 2), nrow = 2), dtype = "f32")
#' nv_solve(a, b)
#' @export
nv_solve <- function(a, b) {
  args <- as_anvl_arrays(a, b)
  a <- args[[1L]]
  b <- args[[2L]]
  L <- prim_cholesky(a, lower = TRUE)
  # Solve L @ y = b
  y <- prim_triangular_solve(L, b, left_side = TRUE, lower = TRUE, unit_diagonal = FALSE, transpose_a = "NO_TRANSPOSE")
  # Solve L^T @ x = y
  prim_triangular_solve(L, y, left_side = TRUE, lower = TRUE, unit_diagonal = FALSE, transpose_a = "TRANSPOSE")
}

#' @title Diagonal Matrix
#' @description
#' Creates a diagonal matrix from a 1-D array.
#' @param operand ([`arrayish`])\cr
#'   A 1-D array of length `n` whose elements become the diagonal entries.
#' @return [`arrayish`]\cr
#'   An `n x n` matrix with `x` on the diagonal and zeros elsewhere.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_diag(nv_array(c(1, 2, 3)))
#' @export
nv_diag <- function(operand) {
  operand <- as_anvl_array(operand)
  n <- shape(operand)[1L]
  zeros <- nv_fill_like(operand, 0, shape = c(n, n))
  idx <- prim_reshape(nv_iota_like(operand, dim = 1L, shape = n, dtype = "i32"), shape = c(n, 1L))
  indices <- nv_concatenate(idx, idx, dimension = 2L)
  prim_scatter(
    zeros,
    indices,
    operand,
    update_window_dims = integer(0),
    inserted_window_dims = c(1L, 2L),
    input_batching_dims = integer(0),
    scatter_indices_batching_dims = integer(0),
    scatter_dims_to_operand_dims = c(1L, 2L),
    index_vector_dim = 2L,
    unique_indices = TRUE
  )
}

#' @title Identity Matrix
#' @description
#' Creates an `n x n` identity matrix.
#'
#' `nv_eye_like()` is a variant where `dtype` and `device` default to those of
#' `like`.
#' @param n (`integer(1)`)\cr
#'   Size of the identity matrix.
#' @param like ([`arrayish`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_eye_like()`).
#' @template param_dtype
#' @template param_device
#' @return [`arrayish`]\cr
#'   An `n x n` identity matrix.
#' @seealso [nv_diag()] for general diagonal matrices.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_eye(3L)
#' x <- nv_array(matrix(0, nrow = 3, ncol = 3), dtype = "f64")
#' nv_eye_like(x, 3L)
#' @export
nv_eye <- function(n, dtype = "f32", device = NULL) {
  nv_diag(nv_fill(1, n, dtype = dtype, device = device))
}

#' @title Sum Reduction
#' @description
#' Sums array elements along the specified dimensions.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce
#' @seealso [prim_reduce_sum()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(1:6, nrow = 2))
#' nv_reduce_sum(x, dims = 1L)
#' @export
nv_reduce_sum <- prim_reduce_sum

#' @title Mean
#' @description
#' Computes the arithmetic mean along the specified dimensions. You can also
#' use `mean()` (which reduces over all dimensions, like base R).
#' @details
#' Implemented as `nv_reduce_sum(operand, dims, drop) / n` where `n` is the
#' product of the reduced dimension sizes.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce
#' @seealso [nv_reduce_sum()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(1:6, nrow = 2))
#' nv_mean(x, dims = 1L)
#' @export
nv_mean <- function(operand, dims, drop = TRUE) {
  operand <- as_anvl_array(operand)
  nelts <- prod(shape(operand)[dims])
  nv_reduce_sum(operand, dims, drop) / nelts
}

#' @title Product Reduction
#' @description
#' Multiplies array elements along the specified dimensions.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce
#' @seealso [prim_reduce_prod()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(1:6, nrow = 2))
#' nv_reduce_prod(x, dims = 1L)
#' @export
nv_reduce_prod <- prim_reduce_prod

#' @title Max Reduction
#' @description
#' Finds the maximum of array elements along the specified dimensions.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce
#' @seealso [prim_reduce_max()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(1:6, nrow = 2))
#' nv_reduce_max(x, dims = 1L)
#' @export
nv_reduce_max <- prim_reduce_max

#' @title Min Reduction
#' @description
#' Finds the minimum of array elements along the specified dimensions.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce
#' @seealso [prim_reduce_min()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(1:6, nrow = 2))
#' nv_reduce_min(x, dims = 1L)
#' @export
nv_reduce_min <- prim_reduce_min

#' @title Any Reduction
#' @description
#' Performs logical OR along the specified dimensions.
#' Returns `TRUE` if any element is `TRUE`.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce_boolean
#' @seealso [prim_reduce_any()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2))
#' nv_reduce_any(x, dims = 1L)
#' @export
nv_reduce_any <- prim_reduce_any

#' @title All Reduction
#' @description
#' Performs logical AND along the specified dimensions.
#' Returns `TRUE` only if all elements are `TRUE`.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce_boolean
#' @seealso [prim_reduce_all()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2))
#' nv_reduce_all(x, dims = 1L)
#' @export
nv_reduce_all <- prim_reduce_all
# Higher order primitives

#' @title Conditional Branching
#' @description
#' Conditional execution of two branches.
#' Unlike [nv_ifelse()], which selects elements, this executes only one
#' of the two branches depending on a scalar predicate.
#' @param pred ([`arrayish`] of boolean type, scalar)\cr
#'   Predicate.
#' @param true (`function()`)\cr
#'   Zero-argument function for the true branch.
#' @param false (`function()`)\cr
#'   Zero-argument function for the false branch.
#'   Must return outputs with the same shapes as the true branch.
#' @return Result of the executed branch.
#' @seealso [prim_if()] for the underlying primitive, [nv_ifelse()] for
#'   element-wise selection.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_if(nv_scalar(TRUE), \() nv_scalar(1), \() nv_scalar(2))
#' @export
nv_if <- prim_if

#' @title While Loop
#' @description
#' Executes a functional while loop.
#' @param init (`list()`)\cr
#'   Named list of initial state values.
#' @param cond (`function`)\cr
#'   Condition function returning a scalar boolean.
#'   Receives the state values as arguments.
#' @param body (`function`)\cr
#'   Body function returning the updated state as a named list
#'   with the same structure as `init`.
#' @return Final state after the loop terminates (same structure as `init`).
#' @seealso [prim_while()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_while(
#'   init = list(i = nv_scalar(0L), total = nv_scalar(0L)),
#'   cond = function(i, total) i < 5L,
#'   body = function(i, total) list(
#'     i = i + 1L,
#'     total = total + i
#'   )
#' )
#' @export
nv_while <- prim_while

## Additional math functions ---------------------------------------------------

#' @title Base-2 Logarithm
#' @description
#' Element-wise base-2 logarithm. You can also use `log2()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nv_log()], [nv_log10()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 4, 8))
#' nv_log2(x)
#' @export
nv_log2 <- function(operand) {
  operand <- as_anvl_array(operand)
  nv_log(operand) / log(2)
}

#' @title Base-10 Logarithm
#' @description
#' Element-wise base-10 logarithm. You can also use `log10()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nv_log()], [nv_log2()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 10, 100, 1000))
#' nv_log10(x)
#' @export
nv_log10 <- function(operand) {
  operand <- as_anvl_array(operand)
  nv_log(operand) / log(10)
}

#' @title Is NaN
#' @description
#' Element-wise check if values are NaN. You can also use `is.nan()`.
#' @template param_operand
#' @template return_unary_boolean
#' @seealso [nv_is_finite()], [nv_is_infinite()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, NaN, Inf, -Inf, 0))
#' nv_is_nan(x)
#' @export
nv_is_nan <- function(operand) {
  operand <- as_anvl_array(operand)
  operand != operand
}

#' @title Is Infinite
#' @description
#' Element-wise check if values are infinite (`Inf` or `-Inf`).
#' You can also use `is.infinite()`.
#' @template param_operand
#' @template return_unary_boolean
#' @seealso [nv_is_finite()], [nv_is_nan()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, NaN, Inf, -Inf, 0))
#' nv_is_infinite(x)
#' @export
nv_is_infinite <- function(operand) {
  operand <- as_anvl_array(operand)
  !nv_is_finite(operand) & (operand == operand)
}

## Reduction operations --------------------------------------------------------

#' @title Variance Reduction
#' @description
#' Computes the variance along the specified dimensions.
#' @details
#' Uses Bessel's correction by default (`correction = 1`), matching R's [var()].
#' Set `correction = 0` for population variance.
#' @template param_operand
#' @template params_reduce
#' @param correction (`integer(1)`)\cr
#'   Degrees of freedom correction. Default is `1` (Bessel's correction).
#' @template return_reduce
#' @seealso [nv_sd()], [nv_mean()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3, 4, 5))
#' nv_var(x, dims = 1L)
#' @export
nv_var <- function(operand, dims, drop = TRUE, correction = 1L) {
  operand <- as_anvl_array(operand)
  assert_int(correction)
  nelts <- prod(shape(operand)[dims])
  mean_bc <- nv_broadcast_to(
    nv_mean(operand, dims, drop = FALSE),
    shape(operand)
  )
  diff <- operand - mean_bc
  nv_reduce_sum(diff * diff, dims, drop) / (nelts - correction)
}

#' @title Standard Deviation Reduction
#' @description
#' Computes the standard deviation along the specified dimensions.
#' @details
#' Uses Bessel's correction by default (`correction = 1`), matching R's [sd()].
#' Set `correction = 0` for population standard deviation.
#' @template param_operand
#' @template params_reduce
#' @param correction (`integer(1)`)\cr
#'   Degrees of freedom correction. Default is `1` (Bessel's correction).
#' @template return_reduce
#' @seealso [nv_var()], [nv_mean()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3, 4, 5))
#' nv_sd(x, dims = 1L)
#' @export
nv_sd <- function(operand, dims, drop = TRUE, correction = 1L) {
  operand <- as_anvl_array(operand)
  nv_sqrt(nv_var(operand, dims, drop, correction))
}

## Array manipulation ----------------------------------------------------------

#' @title Squeeze
#' @description
#' Removes dimensions of size 1 from an array.
#' @template param_operand
#' @param dims (`integer()` | `NULL`)\cr
#'   Dimensions to squeeze. If `NULL` (default), all dimensions of size 1 are removed.
#' @return [`arrayish`]\cr
#'   Has the same data type as `operand` with the specified dimensions removed.
#' @seealso [nv_unsqueeze()], [nv_reshape()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:6, shape = c(1, 6, 1))
#' nv_squeeze(x)
#' @export
nv_squeeze <- function(operand, dims = NULL) {
  operand <- as_anvl_array(operand)
  shp <- shape(operand)
  if (is.null(dims)) {
    new_shape <- shp[shp != 1L]
  } else {
    assert_integerish(dims, lower = 1L, upper = length(shp))
    for (d in dims) {
      if (shp[d] != 1L) {
        cli_abort("Cannot squeeze dimension {d} with size {shp[d]} (must be 1)")
      }
    }
    new_shape <- shp[-dims]
  }
  if (length(new_shape) == 0L) {
    new_shape <- integer(0)
  }
  nv_reshape(operand, new_shape)
}

#' @title Unsqueeze
#' @description
#' Inserts a dimension of size 1 at the specified position.
#' @template param_operand
#' @param dim (`integer(1)`)\cr
#'   Position at which to insert the new dimension.
#' @return [`arrayish`]\cr
#'   Has the same data type as `operand` with an extra dimension of size 1.
#' @seealso [nv_squeeze()], [nv_reshape()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' nv_unsqueeze(x, dim = 1L)
#' @export
nv_unsqueeze <- function(operand, dim) {
  operand <- as_anvl_array(operand)
  shp <- shape(operand)
  assert_int(dim, lower = 1L, upper = length(shp) + 1L)
  new_shape <- append(shp, 1L, after = dim - 1L)
  nv_reshape(operand, new_shape)
}

## Linear algebra --------------------------------------------------------------

#' @title Outer Product
#' @description
#' Computes the outer product of two 1-D arrays.
#' @param x,y ([`arrayish`])\cr
#'   1-D arrays.
#' @return [`arrayish`]\cr
#'   A 2-D array of shape `(length(x), length(y))`.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5))
#' nv_outer(x, y)
#' @export
nv_outer <- function(x, y) {
  args <- nv_promote_to_common(x, y)
  x <- args[[1L]]
  y <- args[[2L]]
  if (ndims(x) != 1L) {
    cli_abort("x must be a 1-D array")
  }
  if (ndims(y) != 1L) {
    cli_abort("y must be a 1-D array")
  }
  x_exp <- nv_unsqueeze(x, dim = 2L)
  y_exp <- nv_unsqueeze(y, dim = 1L)
  bcast <- nv_broadcast_arrays(x_exp, y_exp)
  prim_mul(bcast[[1L]], bcast[[2L]])
}

#' @title Extract Diagonal
#' @description
#' Extracts the diagonal elements from a 2-D array.
#' @template param_operand
#' @return [`arrayish`]\cr
#'   A 1-D array of length `min(nrow, ncol)` containing the diagonal elements.
#' @seealso [nv_diag()] for creating a diagonal matrix, [nv_trace()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:9, shape = c(3, 3))
#' nv_extract_diag(x)
#' @export
nv_extract_diag <- function(operand) {
  operand <- as_anvl_array(operand)
  if (ndims(operand) != 2L) {
    cli_abort("operand must be a 2-D array")
  }
  shp <- shape(operand)
  n <- min(shp)
  idx <- prim_reshape(nv_iota_like(operand, dim = 1L, shape = n, dtype = "i32"), shape = c(n, 1L))
  indices <- nv_concatenate(idx, idx, dimension = 2L)
  prim_gather(
    operand,
    start_indices = indices,
    offset_dims = integer(0),
    collapsed_slice_dims = c(1L, 2L),
    operand_batching_dims = integer(0),
    start_indices_batching_dims = integer(0),
    start_index_map = c(1L, 2L),
    index_vector_dim = 2L,
    slice_sizes = c(1L, 1L)
  )
}

#' @title Matrix Trace
#' @description
#' Computes the trace (sum of diagonal elements) of a 2-D array.
#' @template param_operand
#' @return [`arrayish`]\cr
#'   A scalar with the same data type as `operand`.
#' @seealso [nv_extract_diag()], [nv_diag()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 0, 0, 0, 2, 0, 0, 0, 3), shape = c(3, 3))
#' nv_trace(x)
#' @export
nv_trace <- function(operand) {
  operand <- as_anvl_array(operand)
  diag_vals <- nv_extract_diag(operand)
  nv_reduce_sum(diag_vals, dims = 1L, drop = TRUE)
}

#' @title Lower Triangular Matrix
#' @description
#' Returns the lower triangular part of a 2-D array, setting elements above
#' the specified diagonal to zero.
#' @template param_operand
#' @param diagonal (`integer(1)`)\cr
#'   Diagonal offset. `0` (default) is the main diagonal, positive values
#'   include diagonals above, negative values exclude diagonals below.
#' @return [`arrayish`]\cr
#'   Has the same shape and data type as `operand`.
#' @seealso [nv_triu()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_fill(1, c(3, 3))
#' nv_tril(x)
#' @export
nv_tril <- function(operand, diagonal = 0L) {
  operand <- as_anvl_array(operand)
  if (ndims(operand) != 2L) {
    cli_abort("operand must be a 2-D array")
  }
  assert_int(diagonal)
  rows <- nv_iota_like(operand, dim = 1L, dtype = "i32")
  cols <- nv_iota_like(operand, dim = 2L, dtype = "i32")
  mask <- rows >= cols - as.integer(diagonal)
  nv_ifelse(mask, operand, nv_fill_like(operand, 0))
}

#' @title Upper Triangular Matrix
#' @description
#' Returns the upper triangular part of a 2-D array, setting elements below
#' the specified diagonal to zero.
#' @template param_operand
#' @param diagonal (`integer(1)`)\cr
#'   Diagonal offset. `0` (default) is the main diagonal, positive values
#'   exclude diagonals above, negative values include diagonals below.
#' @return [`arrayish`]\cr
#'   Has the same shape and data type as `operand`.
#' @seealso [nv_tril()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_fill(1, c(3, 3))
#' nv_triu(x)
#' @export
nv_triu <- function(operand, diagonal = 0L) {
  operand <- as_anvl_array(operand)
  if (ndims(operand) != 2L) {
    cli_abort("operand must be a 2-D array")
  }
  assert_int(diagonal)
  rows <- nv_iota_like(operand, dim = 1L, dtype = "i32")
  cols <- nv_iota_like(operand, dim = 2L, dtype = "i32")
  mask <- rows <= cols - as.integer(diagonal)
  nv_ifelse(mask, operand, nv_fill_like(operand, 0))
}

#' @title Cross Product (Matrix)
#' @description
#' Computes `t(x) %*% y`. If `y` is missing, computes `t(x) %*% x`.
#' @param x ([`arrayish`])\cr
#'   An array with at least 2 dimensions.
#' @param y ([`arrayish`] | `NULL`)\cr
#'   Optional second array. If `NULL`, uses `x`.
#' @param ... Unused.
#' @return [`arrayish`]
#' @seealso [nv_tcrossprod()], [nv_matmul()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(1:6, nrow = 3), dtype = "f32")
#' nv_crossprod(x)
#' @export
nv_crossprod <- function(x, y = NULL) {
  if (is.null(y)) {
    x <- as_anvl_array(x)
    y <- x
  } else {
    args <- as_anvl_arrays(x, y)
    x <- args[[1L]]
    y <- args[[2L]]
  }
  nv_matmul(nv_transpose(x), y)
}

#' @title Transpose Cross Product (Matrix)
#' @description
#' Computes `x %*% t(y)`. If `y` is missing, computes `x %*% t(x)`.
#' @param x ([`arrayish`])\cr
#'   An array with at least 2 dimensions.
#' @param y ([`arrayish`] | `NULL`)\cr
#'   Optional second array. If `NULL`, uses `x`.
#' @param ... Unused.
#' @return [`arrayish`]
#' @seealso [nv_crossprod()], [nv_matmul()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(1:6, nrow = 2), dtype = "f32")
#' nv_tcrossprod(x)
#' @export
nv_tcrossprod <- function(x, y = NULL) {
  if (is.null(y)) {
    x <- as_anvl_array(x)
    y <- x
  } else {
    args <- as_anvl_arrays(x, y)
    x <- args[[1L]]
    y <- args[[2L]]
  }
  nv_matmul(x, nv_transpose(y))
}

# Sorting and searching --------------------------------------------------------

# Slice along a single dim at a single (1-based) index, dropping the dim.
.take_at <- function(x, dim, idx) {
  shp <- shape(x)
  start_indices <- rep(1L, length(shp))
  limit_indices <- shp
  strides <- rep(1L, length(shp))
  start_indices[dim] <- idx
  limit_indices[dim] <- idx
  sliced <- prim_static_slice(x, start_indices, limit_indices, strides)
  prim_reshape(sliced, shp[-dim])
}

#' @title Sort
#' @description
#' Sorts an array along a dimension. You can also use `sort()` for 1-D arrays.
#' @param x ([`arrayish`])\cr
#'   The array to sort.
#' @param dim (`integer(1)` | `NULL`)\cr
#'   Dimension along which to sort. If `NULL` (default), uses the last
#'   dimension.
#' @param decreasing (`logical(1)`)\cr
#'   If `TRUE`, sort in decreasing order. Default `FALSE`.
#' @param stable (`logical(1)`)\cr
#'   If `TRUE`, the sort is stable. Default `FALSE`.
#' @return [`arrayish`]\cr
#'   Sorted along `dim`. Same shape and data type as `x`.
#' @seealso [prim_sort()] for the underlying primitive,
#'   [nv_top_k()], [nv_median()], [nv_argmax()], [nv_argmin()].
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
#' nv_sort(x)
#' nv_sort(x, decreasing = TRUE)
#'
#' m <- nv_array(matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE))
#' nv_sort(m, dim = 2L)
#' @export
nv_sort <- function(x, dim = NULL, decreasing = FALSE, stable = FALSE) {
  x <- as_anvl_array(x)
  if (ndims(x) == 0L) {
    cli_abort("Cannot sort a 0-dimensional array")
  }
  dim <- dim %||% ndims(x)
  prim_sort(x, dim = as.integer(dim), descending = decreasing, is_stable = stable)
}

#' @title Argsort
#' @description
#' Returns the indices that would sort the array along a dimension.
#' @param x ([`arrayish`])\cr
#'   The array.
#' @param dim (`integer(1)` | `NULL`)\cr
#'   Dimension along which to compute the sort permutation. If `NULL`
#'   (default), uses the last dimension.
#' @param decreasing (`logical(1)`)\cr
#'   If `TRUE`, indices that produce a decreasing sort. Default `FALSE`.
#' @param stable (`logical(1)`)\cr
#'   If `TRUE`, the sort is stable: indices for equal values keep their
#'   original relative order. Default `FALSE`.
#' @return [`arrayish`] of dtype `i64`\cr
#'   Same shape as `x`. `as_array(x)[as_array(nv_argsort(x))]` reproduces
#'   the sorted array (for 1-D inputs).
#' @seealso [nv_sort()], [prim_sort()].
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(3, 1, 4, 1, 5))
#' nv_argsort(x)
#' @export
nv_argsort <- function(x, dim = NULL, decreasing = FALSE, stable = FALSE) {
  x <- as_anvl_array(x)
  if (ndims(x) == 0L) {
    cli_abort("Cannot argsort a 0-dimensional array")
  }
  dim <- as.integer(dim %||% ndims(x))
  idx <- nv_iota_like(x, dim = dim, dtype = "i64", start = 1L)
  prim_sort(x, idx, dim = dim, descending = decreasing, is_stable = stable)[[2L]]
}

#' @title Top-K Elements
#' @description
#' Returns the `k` largest values along a dimension, sorted in decreasing order.
#' @param x ([`arrayish`])\cr
#'   The array.
#' @param k (`integer(1)`)\cr
#'   Number of top elements to return. Must satisfy `1 <= k <= shape(x)[dim]`.
#' @param dim (`integer(1)` | `NULL`)\cr
#'   Dimension along which to take the top `k`. If `NULL` (default),
#'   uses the last dimension.
#' @return [`arrayish`]\cr
#'   Same shape as `x` except `dim` has size `k`. Values are sorted
#'   in decreasing order along `dim`.
#' @seealso [nv_sort()], [prim_sort()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
#' nv_top_k(x, k = 3L)
#'
#' m <- nv_array(matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE))
#' nv_top_k(m, k = 2L, dim = 2L)
#' @export
nv_top_k <- function(x, k, dim = NULL) {
  x <- as_anvl_array(x)
  if (ndims(x) == 0L) {
    cli_abort("Cannot take top-k of a 0-dimensional array")
  }
  dim <- as.integer(dim %||% ndims(x))
  k <- as.integer(k)
  shp <- shape(x)
  assert_int(k, lower = 1L, upper = shp[dim])
  sorted <- prim_sort(x, dim = dim, descending = TRUE)
  start_indices <- rep(1L, length(shp))
  limit_indices <- shp
  strides <- rep(1L, length(shp))
  limit_indices[dim] <- k
  prim_static_slice(sorted, start_indices, limit_indices, strides)
}

#' @title Median
#' @name nv_median
#' @description
#' Computes the median along a dimension. For an even-length axis, the
#' average of the two middle values is returned (matching base R's
#' `median()`).
#'
#' You can also use `median()` directly on an [`AnvlArray`] or [`AnvlBox`].
#' @param x ([`arrayish`])\cr
#'   The array.
#' @param dim (`integer(1)` | `NULL`)\cr
#'   Dimension along which to compute the median. If `NULL` (default),
#'   uses the last dimension.
#' @param na.rm Included for compatibility with the [stats::median()] generic.
#'   anvl arrays do not carry `NA`s; passing `na.rm = TRUE` raises an error.
#' @param ... Forwarded to `nv_median()`.
#' @return [`arrayish`]\cr
#'   Same shape as `x` with `dim` removed.
#' @seealso [nv_sort()], [prim_sort()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_median(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' median(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' nv_median(nv_array(matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)),
#'   dim = 2L
#' )
#' @export
nv_median <- function(x, dim = NULL) {
  x <- as_anvl_array(x)
  if (ndims(x) == 0L) {
    cli_abort("Cannot compute median of a 0-dimensional array")
  }
  dim <- as.integer(dim %||% ndims(x))
  shp <- shape(x)
  n <- shp[dim]
  sorted <- prim_sort(x, dim = dim)
  if (n %% 2L == 1L) {
    .take_at(sorted, dim, (n + 1L) %/% 2L)
  } else {
    lo <- .take_at(sorted, dim, n %/% 2L)
    hi <- .take_at(sorted, dim, n %/% 2L + 1L)
    nv_div(nv_add(lo, hi), 2)
  }
}

#' @title Index of the Maximum
#' @description
#' Returns the index of the maximum value along a dimension. Ties are broken
#' by returning the smallest index.
#' @param x ([`arrayish`])\cr
#'   The array.
#' @param dim (`integer(1)` | `NULL`)\cr
#'   Dimension along which to find the index. If `NULL` (default), uses
#'   the last dimension.
#' @return [`arrayish`] of dtype `i64`\cr
#'   Same shape as `x` with `dim` removed.
#' @seealso [nv_argmin()], [nv_reduce_max()].
#' @examplesIf pjrt::plugins_downloaded()
#' nv_argmax(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' nv_argmax(nv_array(matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)),
#'   dim = 2L
#' )
#' @export
nv_argmax <- function(x, dim = NULL) {
  .arg_extreme(x, dim, descending = TRUE)
}

#' @title Index of the Minimum
#' @description
#' Returns the index of the minimum value along a dimension. Ties are broken
#' by returning the smallest index.
#' @param x ([`arrayish`])\cr
#'   The array.
#' @param dim (`integer(1)` | `NULL`)\cr
#'   Dimension along which to find the index. If `NULL` (default), uses
#'   the last dimension.
#' @return [`arrayish`] of dtype `i64`\cr
#'   Same shape as `x` with `dim` removed.
#' @seealso [nv_argmax()], [nv_reduce_min()].
#' @examplesIf pjrt::plugins_downloaded()
#' nv_argmin(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' @export
nv_argmin <- function(x, dim = NULL) {
  .arg_extreme(x, dim, descending = FALSE)
}

.arg_extreme <- function(x, dim, descending) {
  x <- as_anvl_array(x)
  if (ndims(x) == 0L) {
    cli_abort("Cannot take arg-extreme of a 0-dimensional array")
  }
  dim <- as.integer(dim %||% ndims(x))
  shp <- shape(x)
  reducer <- if (descending) prim_reduce_max else prim_reduce_min
  extremum <- reducer(x, dims = dim, drop = FALSE)
  extremum <- nv_broadcast_to(extremum, shp)
  is_extremum <- prim_eq(x, extremum)
  idx <- nv_iota_like(x, dim = dim, dtype = "i64", shape = shp, start = 1L)
  big <- nv_fill_like(x, .Machine$integer.max, dtype = "i64", shape = shp)
  selected <- prim_ifelse(is_extremum, idx, big)
  prim_reduce_min(selected, dims = dim, drop = TRUE)
}
