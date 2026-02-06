# This is the user-facing API containing the exported tensor operations.
#' @include primitives.R

# Special tensor creators

#' @title Fill Constant
#' @description
#' Creates a tensor filled with a scalar value. More memory-efficient than
#' `nv_tensor(value, shape = shape)` for large tensors.
#' @param value (`numeric(1)`)\cr
#'   Scalar value to fill the tensor with.
#' @param shape (`integer()`)\cr
#'   Shape of the output tensor.
#' @param dtype (`character(1)` | `NULL`)\cr
#'   Data type. If `NULL` (default), inferred from `value`.
#' @template param_ambiguous
#' @return [`tensorish`]\cr
#'   Has the given `shape` and `dtype`.
#' @seealso [nvl_fill()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval(nv_fill(0, shape = c(2, 3)))
#' @export
nv_fill <- function(value, shape, dtype = NULL, ambiguous = FALSE) {
  dtype <- if (is.null(dtype)) {
    default_dtype(value)
  } else {
    as_dtype(dtype)
  }
  nvl_fill(value, shape, dtype, ambiguous)
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
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   # scalar 1 is broadcast to shape [3]
#'   nv_broadcast_scalars(x, 1)
#' })
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
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(1L)
#'   y <- nv_tensor(1.5)
#'   # integer is promoted to float
#'   nv_promote_to_common(x, y)
#' })
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
#' Broadcasts tensors to a common shape using NumPy-style broadcasting rules.
#'
#' @section Broadcasting Rules:
#' 1. If the tensors have different numbers of dimensions, prepend size-1
#'    dimensions to the shorter shape.
#' 2. For each dimension: if the sizes match, keep them; if one is 1, expand
#'    it to the other's size; otherwise raise an error.
#'
#' @param ... ([`tensorish`])\cr
#'   Tensors to broadcast.
#' @return (`list()` of [`tensorish`])\cr
#'   List of tensors, all with the same shape.
#' @seealso [nv_broadcast_scalars()], [nv_broadcast_to()]
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   y <- nv_tensor(c(10, 20, 30))
#'   nv_broadcast_tensors(x, y)
#' })
#' @export
nv_broadcast_tensors <- function(...) {
  args <- list(...)
  shape <- Reduce(broadcast_shapes, lapply(args, shape_abstract))
  lapply(args, nv_broadcast_to, shape = shape)
}

#' @title Broadcast to Shape
#' @description
#' Broadcasts a tensor to a target shape using NumPy-style broadcasting rules.
#' @template param_operand
#' @param shape (`integer()`)\cr
#'   Target shape. Each existing dimension must either match or be 1.
#' @return [`tensorish`]\cr
#'   Has the given `shape` and the same data type as `operand`.
#' @seealso [nv_broadcast_tensors()], [nv_broadcast_scalars()],
#'   [nvl_broadcast_in_dim()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   nv_broadcast_to(x, shape = c(2, 3))
#' })
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

#' @title Convert Data Type
#' @description
#' Converts the elements of a tensor to a different data type.
#' Returns the input unchanged if it already has the target type.
#' @template param_operand
#' @template param_dtype
#' @return [`tensorish`]\cr
#'   Has the given `dtype` and the same shape as `operand`.
#' @seealso [nvl_convert()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1L, 2L, 3L))
#'   nv_convert(x, dtype = "f32")
#' })
#' @export
nv_convert <- function(operand, dtype) {
  if (dtype_abstract(operand) != as_dtype(dtype)) {
    nvl_convert(operand, dtype = as_dtype(dtype), ambiguous = FALSE)
  } else {
    operand
  }
}

#' @rdname nv_transpose
#' @export
nv_transpose <- function(x, permutation = NULL) {
  permutation <- permutation %??% rev(seq_len(ndims_abstract(x)))
  nvl_transpose(x, permutation)
}


#' @title Reshape
#' @description
#' Reshapes a tensor to a new shape without changing the underlying data.
#' Returns the input unchanged if it already has the target shape.
#' @details
#' Note that row-major order is used, which differs from R's column-major order.
#' @template param_operand
#' @param shape (`integer()`)\cr
#'   Target shape. Must have the same number of elements as `operand`.
#' @return [`tensorish`]\cr
#'   Has the given `shape` and the same data type as `operand`.
#' @seealso [nvl_reshape()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(1:6)
#'   nv_reshape(x, c(2, 3))
#' })
#' @export
nv_reshape <- function(operand, shape) {
  if (!identical(shape_abstract(operand), shape)) {
    nvl_reshape(operand, shape)
  } else {
    operand
  }
}

#' @title Concatenate
#' @description
#' Concatenates tensors along a dimension. Operands are promoted to a common
#' data type and scalars are broadcast before concatenation.
#' @param ... ([`tensorish`])\cr
#'   Tensors to concatenate. Must have the same shape except along `dimension`.
#' @param dimension (`integer(1)` | `NULL`)\cr
#'   Dimension along which to concatenate.
#'   If `NULL` (default), assumes all inputs are at most 1-D and concatenates along dimension 1.
#' @return [`tensorish`]\cr
#'   Has the common data type and a shape matching the inputs in all
#'   dimensions except `dimension`, which is the sum of input sizes.
#' @seealso [nvl_concatenate()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(4, 5, 6))
#'   nv_concatenate(x, y)
#' })
#' @export
nv_concatenate <- function(..., dimension = NULL) {
  args <- list(...)
  args <- do.call(nv_promote_to_common, args)
  shapes <- lapply(args, shape_abstract)
  ranks <- lengths(shapes)
  non_scalar_shapes <- shapes[ranks > 0L]
  n_scalars <- sum(ranks == 0L)
  assert_int(dimension, lower = 1L, upper = max(max(ranks), 1L), null.ok = max(ranks) <= 1L)
  dimension <- dimension %??% 1L

  non_scalar_shapes_without_dim <- lapply(non_scalar_shapes, \(shape) {
    shape[-dimension]
  })
  if (length(non_scalar_shapes) && length(unique(non_scalar_shapes_without_dim)) != 1L) {
    cli_abort(c(
      "All non-scalar tensors must have the same shape (except for the concatenation dimension)",
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
    if (ndims_abstract(arg) == 0L) {
      nv_broadcast_to(arg, out_shape_dim_is_one)
    } else {
      arg
    }
  })
  rlang::exec(nvl_concatenate, !!!args, dimension = dimension)
}

#' @title Static Slice
#' @description
#' Extracts a slice from a tensor using static (compile-time) indices.
#' For dynamic indexing, use [nv_subset()] instead.
#' @template param_operand
#' @param start_indices (`integer()`)\cr
#'   Start indices (inclusive), one per dimension.
#' @param limit_indices (`integer()`)\cr
#'   End indices (inclusive), one per dimension.
#' @param strides (`integer()`)\cr
#'   Step sizes, one per dimension. A stride of 1 selects every element.
#' @return [`tensorish`]\cr
#'   Has the same data type as `operand`.
#' @seealso [nv_subset()], [nvl_static_slice()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(1:10)
#'   nv_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 1L)
#' })
#' @export
nv_static_slice <- nvl_static_slice

#' @title Print Tensor
#' @description
#' Prints a tensor value to the console during JIT execution and returns the
#' input unchanged. Useful for debugging.
#' @note Currently only works on the CPU backend.
#' @template param_operand
#' @return [`tensorish`]\cr
#'   Returns `operand` unchanged.
#' @seealso [nvl_print()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3), device = "cpu")
#'   nv_print(x)
#' })
#' @export
nv_print <- nvl_print

#' @title Conditional Element Selection
#' @description
#' Selects elements from `true_value` or `false_value` based on `pred`,
#' analogous to R's [ifelse()].
#' @param pred ([`tensorish`] of boolean type)\cr
#'   Predicate tensor. Must be scalar or the same shape as `true_value`.
#' @param true_value ([`tensorish`])\cr
#'   Values to return where `pred` is `TRUE`.
#' @param false_value ([`tensorish`])\cr
#'   Values to return where `pred` is `FALSE`.
#'   Must have the same shape and data type as `true_value`.
#' @return [`tensorish`]\cr
#'   Has the same shape and data type as `true_value`.
#' @seealso [nvl_ifelse()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   pred <- nv_tensor(c(TRUE, FALSE, TRUE))
#'   nv_ifelse(pred, nv_tensor(c(1, 2, 3)), nv_tensor(c(4, 5, 6)))
#' })
#' @export
nv_ifelse <- nvl_ifelse

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
#' Adds two tensors element-wise. You can also use the `+` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_add()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(4, 5, 6))
#'   x + y
#' })
#' @export
nv_add <- make_do_binary(nvl_add)

#' @title Multiplication
#' @description
#' Multiplies two tensors element-wise. You can also use the `*` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_mul()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(4, 5, 6))
#'   x * y
#' })
#' @export
nv_mul <- make_do_binary(nvl_mul)

#' @title Subtraction
#' @description
#' Subtracts two tensors element-wise. You can also use the `-` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_sub()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(4, 5, 6))
#'   y <- nv_tensor(c(1, 2, 3))
#'   x - y
#' })
#' @export
nv_sub <- make_do_binary(nvl_sub)

#' @title Division
#' @description
#' Divides two tensors element-wise. You can also use the `/` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_div()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(10, 20, 30))
#'   y <- nv_tensor(c(2, 5, 10))
#'   x / y
#' })
#' @export
nv_div <- make_do_binary(nvl_div)

#' @title Power
#' @description
#' Raises `lhs` to the power of `rhs` element-wise. You can also use the `^` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_pow()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(2, 3, 4))
#'   y <- nv_tensor(c(3, 2, 1))
#'   x ^ y
#' })
#' @export
nv_pow <- make_do_binary(nvl_pow)

#' @title Equal
#' @description
#' Element-wise equality comparison. You can also use the `==` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [nvl_eq()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(1, 3, 2))
#'   x == y
#' })
#' @export
nv_eq <- make_do_binary(nvl_eq)

#' @title Not Equal
#' @description
#' Element-wise inequality comparison. You can also use the `!=` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [nvl_ne()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(1, 3, 2))
#'   x != y
#' })
#' @export
nv_ne <- make_do_binary(nvl_ne)

#' @title Greater Than
#' @description
#' Element-wise greater than comparison. You can also use the `>` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [nvl_gt()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(3, 2, 1))
#'   x > y
#' })
#' @export
nv_gt <- make_do_binary(nvl_gt)

#' @title Greater Than or Equal
#' @description
#' Element-wise greater than or equal comparison. You can also use the `>=` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [nvl_ge()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(3, 2, 1))
#'   x >= y
#' })
#' @export
nv_ge <- make_do_binary(nvl_ge)

#' @title Less Than
#' @description
#' Element-wise less than comparison. You can also use the `<` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [nvl_lt()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(3, 2, 1))
#'   x < y
#' })
#' @export
nv_lt <- make_do_binary(nvl_lt)

#' @title Less Than or Equal
#' @description
#' Element-wise less than or equal comparison. You can also use the `<=` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [nvl_le()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(3, 2, 1))
#'   x <= y
#' })
#' @export
nv_le <- make_do_binary(nvl_le)

#' @title Maximum
#' @description
#' Element-wise maximum of two tensors.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_max()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 5, 3))
#'   y <- nv_tensor(c(4, 2, 6))
#'   nv_max(x, y)
#' })
#' @export
nv_max <- make_do_binary(nvl_max)

#' @title Minimum
#' @description
#' Element-wise minimum of two tensors.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_min()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 5, 3))
#'   y <- nv_tensor(c(4, 2, 6))
#'   nv_min(x, y)
#' })
#' @export
nv_min <- make_do_binary(nvl_min)

#' @title Remainder
#' @description
#' Element-wise remainder of division. You can also use the `%%` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_remainder()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(7, 8, 9))
#'   y <- nv_tensor(c(3, 3, 4))
#'   x %% y
#' })
#' @export
nv_remainder <- make_do_binary(nvl_remainder)

#' @title Logical And
#' @description
#' Element-wise logical AND. You can also use the `&` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_and()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(TRUE, FALSE, TRUE))
#'   y <- nv_tensor(c(TRUE, TRUE, FALSE))
#'   x & y
#' })
#' @export
nv_and <- make_do_binary(nvl_and)

#' @title Logical Or
#' @description
#' Element-wise logical OR. You can also use the `|` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_or()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(TRUE, FALSE, TRUE))
#'   y <- nv_tensor(c(TRUE, TRUE, FALSE))
#'   x | y
#' })
#' @export
nv_or <- make_do_binary(nvl_or)

#' @title Logical Xor
#' @description
#' Element-wise logical XOR.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_xor()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(TRUE, FALSE, TRUE))
#'   y <- nv_tensor(c(TRUE, TRUE, FALSE))
#'   nv_xor(x, y)
#' })
#' @export
nv_xor <- make_do_binary(nvl_xor)

#' @title Shift Left
#' @description
#' Element-wise left bit shift.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_shift_left()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1L, 2L, 4L))
#'   y <- nv_tensor(c(1L, 2L, 1L))
#'   nv_shift_left(x, y)
#' })
#' @export
nv_shift_left <- make_do_binary(nvl_shift_left)

#' @title Logical Shift Right
#' @description
#' Element-wise logical right bit shift.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_shift_right_logical()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(8L, 16L, 32L))
#'   y <- nv_tensor(c(1L, 2L, 3L))
#'   nv_shift_right_logical(x, y)
#' })
#' @export
nv_shift_right_logical <- make_do_binary(nvl_shift_right_logical)

#' @title Arithmetic Shift Right
#' @description
#' Element-wise arithmetic right bit shift.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_shift_right_arithmetic()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(8L, -16L, 32L))
#'   y <- nv_tensor(c(1L, 2L, 3L))
#'   nv_shift_right_arithmetic(x, y)
#' })
#' @export
nv_shift_right_arithmetic <- make_do_binary(nvl_shift_right_arithmetic)

#' @title Arctangent 2
#' @description
#' Element-wise two-argument arctangent, i.e. the angle (in radians) between the positive
#' x-axis and the point `(rhs, lhs)`.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nvl_atan2()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   y <- nv_tensor(c(1, 0, -1))
#'   x <- nv_tensor(c(0, 1, 0))
#'   nv_atan2(y, x)
#' })
#' @export
nv_atan2 <- make_do_binary(nvl_atan2)


#' @title Bitcast Conversion
#' @name nv_bitcast_convert
#' @description
#' Reinterprets the bits of a tensor as a different data type without modifying
#' the underlying data. If the target type is narrower, an extra trailing
#' dimension is added; if wider, the last dimension is consumed.
#' @template param_operand
#' @param dtype (`character(1)` | [`TensorDataType`])\cr
#'   Target data type.
#' @return [`tensorish`]\cr
#'   Has the given `dtype`.
#' @seealso [nvl_bitcast_convert()] for the underlying primitive, [nv_convert()]
#'   for value-preserving type conversion.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(1L)
#'   nvl_bitcast_convert(x, dtype = "i8")
#' })
#' @export
nv_bitcast_convert <- nvl_bitcast_convert

## Unary ops ------------------------------------------------------------------

#' @title Negation
#' @description
#' Negates a tensor element-wise. You can also use the unary `-` operator.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_negate()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, -2, 3))
#'   -x
#' })
#' @export
nv_negate <- nvl_negate

#' @title Logical Not
#' @description
#' Element-wise logical NOT. You can also use the `!` operator.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_not()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(TRUE, FALSE, TRUE))
#'   !x
#' })
#' @export
nv_not <- nvl_not

#' @title Absolute Value
#' @description
#' Element-wise absolute value. You can also use `abs()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_abs()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(-1, 2, -3))
#'   abs(x)
#' })
#' @export
nv_abs <- nvl_abs

#' @title Square Root
#' @description
#' Element-wise square root. You can also use `sqrt()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_sqrt()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 4, 9))
#'   sqrt(x)
#' })
#' @export
nv_sqrt <- nvl_sqrt

#' @title Reciprocal Square Root
#' @description
#' Element-wise reciprocal square root, i.e. `1 / sqrt(x)`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_rsqrt()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 4, 9))
#'   nv_rsqrt(x)
#' })
#' @export
nv_rsqrt <- nvl_rsqrt

#' @title Natural Logarithm
#' @description
#' Element-wise natural logarithm. You can also use `log()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_log()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2.718, 7.389))
#'   log(x)
#' })
#' @export
nv_log <- nvl_log

#' @title Hyperbolic Tangent
#' @description
#' Element-wise hyperbolic tangent. You can also use `tanh()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_tanh()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(-1, 0, 1))
#'   tanh(x)
#' })
#' @export
nv_tanh <- nvl_tanh

#' @title Tangent
#' @description
#' Element-wise tangent. You can also use `tan()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_tan()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, 0.5, 1))
#'   tan(x)
#' })
#' @export
nv_tan <- nvl_tan

#' @title Sine
#' @description
#' Element-wise sine. You can also use `sin()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_sine()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, pi / 2, pi))
#'   sin(x)
#' })
#' @export
nv_sine <- nvl_sine

#' @title Cosine
#' @description
#' Element-wise cosine. You can also use `cos()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_cosine()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, pi / 2, pi))
#'   cos(x)
#' })
#' @export
nv_cosine <- nvl_cosine

#' @title Floor
#' @description
#' Element-wise floor (round toward negative infinity). You can also use `floor()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_floor()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1.2, 2.7, -1.5))
#'   floor(x)
#' })
#' @export
nv_floor <- nvl_floor

#' @title Ceiling
#' @description
#' Element-wise ceiling (round toward positive infinity). You can also use `ceiling()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_ceil()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1.2, 2.7, -1.5))
#'   ceiling(x)
#' })
#' @export
nv_ceil <- nvl_ceil

#' @title Sign
#' @description
#' Element-wise sign function. You can also use `sign()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_sign()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(-3, 0, 5))
#'   sign(x)
#' })
#' @export
nv_sign <- nvl_sign

#' @title Exponential
#' @description
#' Element-wise exponential. You can also use `exp()`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_exp()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, 1, 2))
#'   exp(x)
#' })
#' @export
nv_exp <- nvl_exp

#' @title Exponential Minus One
#' @description
#' Element-wise `exp(x) - 1`, more accurate for small `x`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_expm1()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, 0.001, 1))
#'   nv_expm1(x)
#' })
#' @export
nv_expm1 <- nvl_expm1

#' @title Log Plus One
#' @description
#' Element-wise `log(1 + x)`, more accurate for small `x`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_log1p()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, 0.001, 1))
#'   nv_log1p(x)
#' })
#' @export
nv_log1p <- nvl_log1p

#' @title Cube Root
#' @description
#' Element-wise cube root.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_cbrt()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 8, 27))
#'   nv_cbrt(x)
#' })
#' @export
nv_cbrt <- nvl_cbrt

#' @title Logistic (Sigmoid)
#' @description
#' Element-wise logistic sigmoid: `1 / (1 + exp(-x))`.
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_logistic()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(-2, 0, 2))
#'   nv_logistic(x)
#' })
#' @export
nv_logistic <- nvl_logistic

#' @title Is Finite
#' @description
#' Element-wise check if values are finite (not `Inf`, `-Inf`, or `NaN`).
#' @template param_operand
#' @template return_unary_boolean
#' @seealso [nvl_is_finite()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, Inf, NaN, -Inf, 0))
#'   nv_is_finite(x)
#' })
#' @export
nv_is_finite <- nvl_is_finite

#' @title Population Count
#' @description
#' Element-wise population count (number of set bits).
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_popcnt()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(7L, 3L, 15L))
#'   nv_popcnt(x)
#' })
#' @export
nv_popcnt <- nvl_popcnt

#' @title Clamp
#' @description
#' Element-wise clamp: `max(min_val, min(operand, max_val))`.
#' Converts `min_val` and `max_val` to the data type of `operand`.
#' @details
#' The underlying stableHLO function already broadcasts scalars, so no need to broadcast manually.
#' @param min_val,max_val ([`tensorish`])\cr
#'   Minimum and maximum values (scalar or same shape as `operand`).
#' @template param_operand
#' @template return_unary
#' @seealso [nvl_clamp()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(-1, 0.5, 2))
#'   nv_clamp(nv_scalar(0), x, nv_scalar(1))
#' })
#' @export
nv_clamp <- function(min_val, operand, max_val) {
  op_dtype <- dtype_abstract(operand)
  min_val <- nv_convert(min_val, op_dtype)
  max_val <- nv_convert(max_val, op_dtype)
  nvl_clamp(min_val, operand, max_val)
}

#' @title Reverse
#' @description
#' Reverses the order of elements along specified dimensions.
#' @template param_operand
#' @param dims (`integer()`)\cr
#'   Dimensions to reverse.
#' @return [`tensorish`]\cr
#'   Has the same shape and data type as `operand`.
#' @seealso [nvl_reverse()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3, 4, 5))
#'   nv_reverse(x, dims = 1L)
#' })
#' @export
nv_reverse <- nvl_reverse

#' @title Iota
#' @description
#' Creates a tensor with values increasing along the specified dimension,
#' starting from `start`.
#' @param dim (`integer(1)`)\cr
#'   Dimension along which values increase.
#' @template param_dtype
#' @template param_shape
#' @param start (`integer(1)`)\cr
#'   Starting value (default 1).
#' @template param_ambiguous
#' @return [`tensorish`]\cr
#'   Has the given `dtype` and `shape`.
#' @seealso [nv_seq()] for a simpler 1-D sequence, [nvl_iota()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval(nv_iota(dim = 1L, dtype = "i32", shape = 5L))
#' @export
nv_iota <- nvl_iota

#' @title Sequence
#' @description
#' Creates a 1-D tensor with integer values from `start` to `end` (inclusive),
#' analogous to R's `seq(start, end)`.
#' @param start,end (`integer(1)`)\cr
#'   Start and end values. Must satisfy `start <= end`.
#' @template param_dtype
#' @template param_ambiguous
#' @return [`tensorish`]\cr
#'   1-D tensor of length `end - start + 1`.
#' @seealso [nv_iota()] for multi-dimensional sequences.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval(nv_seq(3, 7))
#' @export
nv_seq <- function(start, end, dtype = "i32", ambiguous = FALSE) {
  assert_int(start)
  assert_int(end)
  assert(start <= end)
  nv_iota(shape = end - start + 1, dtype = dtype, ambiguous = ambiguous, dim = 1L, start = start)
}

#' @title Pad
#' @description
#' Pads a tensor with a given value at the edges and optionally between elements.
#' @template param_operand
#' @param padding_value ([`tensorish`])\cr
#'   Scalar value to use for padding. Must have the same dtype as `operand`.
#' @param edge_padding_low (`integer()`)\cr
#'   Amount of padding to add at the start of each dimension.
#' @param edge_padding_high (`integer()`)\cr
#'   Amount of padding to add at the end of each dimension.
#' @param interior_padding (`integer()` | `NULL`)\cr
#'   Amount of padding to add between elements in each dimension.
#'   If `NULL` (default), no interior padding is applied.
#' @return [`tensorish`]\cr
#'   Has the same data type as `operand`.
#' @seealso [nvl_pad()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   nv_pad(x, nv_scalar(0), edge_padding_low = 2L, edge_padding_high = 1L)
#' })
#' @export
nv_pad <- function(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding = NULL) {
  rank <- ndims_abstract(operand)
  if (is.null(interior_padding)) {
    interior_padding <- rep(0L, rank)
  }
  nvl_pad(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding)
}

#' @title Round
#' @description
#' Element-wise rounding. You can also use the `round()` generic.
#' @template param_operand
#' @param method (`character(1)`)\cr
#'   Rounding method.
#'   Either `"nearest_even"` (default) or `"afz"` (away from zero).
#' @template return_unary
#' @seealso [nvl_round()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1.4, 2.5, 3.6))
#'   round(x)
#' })
#' @export
nv_round <- nvl_round

## Other operations -----------------------------------------------------------

#' @title Matrix Multiplication
#' @description
#' Matrix multiplication of two tensors. You can also use the `%*%` operator.
#' Supports batched matrix multiplication when inputs have more than 2 dimensions.
#' @section Shapes:
#' - `lhs`: `(b1, ..., bk, m, n)`
#' - `rhs`: `(b1, ..., bk, n, p)`
#' - output: `(b1, ..., bk, m, p)`
#' @param lhs,rhs ([`tensorish`])\cr
#'   Tensors with at least 2 dimensions.
#'   Operands are [promoted to a common data type][nv_promote_to_common()].
#' @return [`tensorish`]
#' @seealso [nvl_dot_general()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   y <- nv_tensor(matrix(1:6, nrow = 3))
#'   x %*% y
#' })
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
  nbatch_lhs <- ndims_abstract(lhs) - 2L
  nbatch_rhs <- ndims_abstract(rhs) - 2L
  nbatch <- min(nbatch_lhs, nbatch_rhs)
  lhs_batch <- if (nbatch > 0L) {
    (nbatch_lhs - nbatch + 1L):nbatch_lhs
  } else {
    integer()
  }
  rhs_batch <- if (nbatch > 0L) {
    (nbatch_rhs - nbatch + 1L):nbatch_rhs
  } else {
    integer()
  }
  nvl_dot_general(
    lhs,
    rhs,
    contracting_dims = list(ndims_abstract(lhs), ndims_abstract(rhs) - 1L),
    batching_dims = list(lhs_batch, rhs_batch)
  )
}

#' @title Sum Reduction
#' @description
#' Sums tensor elements along the specified dimensions.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce
#' @seealso [nvl_reduce_sum()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   nv_reduce_sum(x, dims = 1L)
#' })
#' @export
nv_reduce_sum <- nvl_reduce_sum

#' @title Mean Reduction
#' @description
#' Computes the arithmetic mean along the specified dimensions.
#' @details
#' Implemented as `nv_reduce_sum(operand, dims, drop) / n` where `n` is the
#' product of the reduced dimension sizes.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce
#' @seealso [nv_reduce_sum()]
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   nv_reduce_mean(x, dims = 1L)
#' })
#' @export
nv_reduce_mean <- function(operand, dims, drop = TRUE) {
  # TODO: division by zero?
  nelts <- prod(shape_abstract(operand)[dims])
  nv_reduce_sum(operand, dims, drop) / nelts
}

#' @title Product Reduction
#' @description
#' Multiplies tensor elements along the specified dimensions.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce
#' @seealso [nvl_reduce_prod()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   nv_reduce_prod(x, dims = 1L)
#' })
#' @export
nv_reduce_prod <- nvl_reduce_prod

#' @title Max Reduction
#' @description
#' Finds the maximum of tensor elements along the specified dimensions.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce
#' @seealso [nvl_reduce_max()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   nv_reduce_max(x, dims = 1L)
#' })
#' @export
nv_reduce_max <- nvl_reduce_max

#' @title Min Reduction
#' @description
#' Finds the minimum of tensor elements along the specified dimensions.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce
#' @seealso [nvl_reduce_min()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   nv_reduce_min(x, dims = 1L)
#' })
#' @export
nv_reduce_min <- nvl_reduce_min

#' @title Any Reduction
#' @description
#' Performs logical OR along the specified dimensions.
#' Returns `TRUE` if any element is `TRUE`.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce_boolean
#' @seealso [nvl_reduce_any()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2))
#'   nv_reduce_any(x, dims = 1L)
#' })
#' @export
nv_reduce_any <- nvl_reduce_any

#' @title All Reduction
#' @description
#' Performs logical AND along the specified dimensions.
#' Returns `TRUE` only if all elements are `TRUE`.
#' @template param_operand
#' @template params_reduce
#' @template return_reduce_boolean
#' @seealso [nvl_reduce_all()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2))
#'   nv_reduce_all(x, dims = 1L)
#' })
#' @export
nv_reduce_all <- nvl_reduce_all
# Higher order primitives

#' @title Conditional Branching
#' @description
#' Conditional execution of two branches.
#' Unlike [nv_ifelse()], which selects elements, this executes only one
#' of the two branches depending on a scalar predicate.
#' @param pred ([`tensorish`] of boolean type, scalar)\cr
#'   Predicate.
#' @param true (`expression`)\cr
#'   Expression for the true branch (non-standard evaluation).
#' @param false (`expression`)\cr
#'   Expression for the false branch (non-standard evaluation).
#'   Must return outputs with the same shapes as the true branch.
#' @return Result of the executed branch.
#' @seealso [nvl_if()] for the underlying primitive, [nv_ifelse()] for
#'   element-wise selection.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval(nv_if(nv_scalar(TRUE), nv_scalar(1), nv_scalar(2)))
#' @export
nv_if <- nvl_if

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
#' @seealso [nvl_while()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   nv_while(
#'     init = list(i = nv_scalar(0L), total = nv_scalar(0L)),
#'     cond = function(i, total) i < 5L,
#'     body = function(i, total) list(
#'       i = i + 1L,
#'       total = total + i
#'     )
#'   )
#' })
#' @export
nv_while <- nvl_while
