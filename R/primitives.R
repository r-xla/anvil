#' @include utils.R
#' @include type-converters.R
#' @include primitive.R

make_binary_op <- function(prim, stablehlo_infer) {
  force(stablehlo_infer)
  infer_fn <- function(lhs, rhs) {
    both_ambiguous <- lhs$ambiguous && rhs$ambiguous
    out <- stablehlo_infer(at2vt(lhs), at2vt(rhs))[[1L]]
    out <- vt2at(out)
    out$ambiguous <- both_ambiguous
    list(out)
  }
  function(lhs, rhs) {
    graph_desc_add(prim, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
  }
}

make_unary_op <- function(prim, stablehlo_infer) {
  force(stablehlo_infer)
  infer_fn <- function(operand) {
    out <- stablehlo_infer(at2vt(operand))[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  function(operand) {
    graph_desc_add(prim, list(operand = operand), infer_fn = infer_fn)[[1L]]
  }
}


infer_reduce <- function(operand, dims, drop) {
  old_shape <- shape(operand)
  if (drop) {
    new_shape <- old_shape[-dims]
  } else {
    new_shape <- old_shape
    new_shape[dims] <- 1L
  }
  list(AbstractTensor(
    dtype = dtype(operand),
    shape = Shape(new_shape),
    ambiguous = operand$ambiguous
  ))
}

infer_reduce_boolean <- function(operand, dims, drop) {
  old_shape <- shape(operand)
  if (drop) {
    new_shape <- old_shape[-dims]
  } else {
    new_shape <- old_shape
    new_shape[dims] <- 1L
  }
  list(AbstractTensor(
    dtype = "pred",
    shape = Shape(new_shape),
    ambiguous = operand$ambiguous
  ))
}

p_fill <- AnvilPrimitive("fill")
#' @title Primitive Fill
#' @description
#' Creates a tensor filled with a scalar value.
#' @param value (`numeric(1)`)\cr
#'   Scalar value.
#' @template param_shape
#' @template param_dtype
#' @template param_ambiguous
#' @return [`tensorish`]
#' @export
nvl_fill <- function(value, shape, dtype, ambiguous = FALSE) {
  infer_fill <- function(value, shape, dtype, ambiguous) {
    list(AbstractTensor(dtype = as_dtype(dtype), shape = shape, ambiguous = ambiguous))
  }
  graph_desc_add(
    p_fill,
    list(),
    params = list(value = value, dtype = dtype, shape = shape, ambiguous = ambiguous),
    infer_fn = infer_fill
  )[[1L]]
}

p_add <- AnvilPrimitive("add")
#' @title Primitive Addition
#' @description
#' Adds two tensors element-wise.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_add <- make_binary_op(p_add, stablehlo::infer_types_add)

p_mul <- AnvilPrimitive("mul")
#' @title Primitive Multiplication
#' @description
#' Multiplies two tensors element-wise.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_mul <- make_binary_op(p_mul, stablehlo::infer_types_multiply)

p_sub <- AnvilPrimitive("sub")
#' @title Primitive Subtraction
#' @description
#' Subtracts two tensors element-wise.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_sub <- make_binary_op(p_sub, stablehlo::infer_types_subtract)

p_negate <- AnvilPrimitive("negate")
#' @title Primitive Negation
#' @description
#' Negates a tensor element-wise.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_negate <- make_unary_op(p_negate, stablehlo::infer_types_negate)

p_div <- AnvilPrimitive("divide")
#' @title Primitive Division
#' @description
#' Divides two tensors element-wise.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_div <- make_binary_op(p_div, stablehlo::infer_types_divide)

p_pow <- AnvilPrimitive("power")
#' @title Primitive Power
#' @description
#' Raises lhs to the power of rhs element-wise.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_pow <- make_binary_op(p_pow, stablehlo::infer_types_power)

p_broadcast_in_dim <- AnvilPrimitive("broadcast_in_dim")
#' @title Primitive Broadcast
#' @description
#' Broadcasts a tensor to a new shape.
#' @template param_operand
#' @param shape (`integer()`)\cr
#'   Target shape.
#' @param broadcast_dimensions (`integer()`)\cr
#'   Dimension mapping.
#' @return [`tensorish`]
#' @importFrom stablehlo r_to_constant
#' @export
nvl_broadcast_in_dim <- function(operand, shape, broadcast_dimensions) {
  infer_fn <- function(operand, shape, broadcast_dimensions) {
    bd_attr <- r_to_constant(
      as.integer(broadcast_dimensions - 1L),
      dtype = "i64",
      shape = length(broadcast_dimensions)
    )
    out <- stablehlo::infer_types_broadcast_in_dim(
      at2vt(operand),
      broadcast_dimensions = bd_attr,
      shape = shape
    )[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  graph_desc_add(
    p_broadcast_in_dim,
    list(operand = operand),
    params = list(
      shape = shape,
      broadcast_dimensions = broadcast_dimensions
    ),
    infer_fn = infer_fn
  )[[1L]]
}

p_dot_general <- AnvilPrimitive("dot_general")
#' @title Primitive Dot General
#' @description
#' General dot product of two tensors.
#' @template params_lhs_rhs
#' @param contracting_dims (`list()`)\cr
#'   Dimensions to contract.
#' @param batching_dims (`list()`)\cr
#'   Batch dimensions.
#' @return [`tensorish`]
#' @export
nvl_dot_general <- function(lhs, rhs, contracting_dims, batching_dims) {
  infer_fn <- function(lhs, rhs, contracting_dims, batching_dims) {
    ddn <- stablehlo::DotDimensionNumbers(
      contracting_dims = lapply(contracting_dims, \(x) x - 1L),
      batching_dims = lapply(batching_dims, \(x) x - 1L)
    )
    out <- stablehlo::infer_types_dot_general(at2vt(lhs), at2vt(rhs), dot_dimension_numbers = ddn)[[1L]]
    list(vt2at(out))
  }
  graph_desc_add(
    p_dot_general,
    list(lhs = lhs, rhs = rhs),
    list(contracting_dims = contracting_dims, batching_dims = batching_dims),
    infer_fn = infer_fn
  )[[1L]]
}

p_transpose <- AnvilPrimitive("transpose")
#' @title Primitive Transpose
#' @description
#' Transposes a tensor according to a permutation.
#' @template param_operand
#' @param permutation (`integer()`)\cr
#'   Dimension permutation.
#' @return [`tensorish`]
#' @export
nvl_transpose <- function(operand, permutation) {
  infer_fn <- function(operand, permutation) {
    perm_attr <- r_to_constant(
      as.integer(permutation - 1L),
      dtype = "i64",
      shape = length(permutation)
    )
    out <- stablehlo::infer_types_transpose(at2vt(operand), permutation = perm_attr)[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  graph_desc_add(
    p_transpose,
    list(operand = operand),
    list(permutation = permutation),
    infer_fn = infer_fn
  )[[1L]]
}

p_reshape <- AnvilPrimitive("reshape")
#' @title Primitive Reshape
#' @description
#' Reshapes a tensor to a new shape.
#' @template param_operand
#' @template param_shape
#' @return [`tensorish`]
#' @export
nvl_reshape <- function(operand, shape) {
  infer_fn <- function(operand, shape) {
    out <- stablehlo::infer_types_reshape(at2vt(operand), shape = shape)[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  graph_desc_add(
    p_reshape,
    list(operand = operand),
    params = list(shape = shape),
    infer_fn = infer_fn
  )[[1L]]
}

p_concatenate <- AnvilPrimitive("concatenate")
#' @title Primitive Concatenate
#' @description
#' Concatenates tensors along a dimension.
#' @param ... ([`tensorish`])\cr
#'   Tensors to concatenate.
#' @param dimension (`integer(1)`)\cr
#'   Dimension to concatenate along.
#' @return [`tensorish`]
#' @export
nvl_concatenate <- function(..., dimension) {
  dots <- list(...)
  infer_fn <- function(..., dimension) {
    operands <- list(...)
    all_ambiguous <- all(vapply(operands, \(x) x$ambiguous, logical(1L)))
    vts <- lapply(operands, at2vt)
    # Convert dimension to Constant as required by stablehlo
    dim_const <- stablehlo::r_to_constant(
      as.integer(dimension - 1L),
      dtype = "i64",
      shape = integer(0)
    )
    out <- rlang::exec(stablehlo::infer_types_concatenate, !!!vts, dimension = dim_const)[[1L]]
    out <- vt2at(out)
    out$ambiguous <- all_ambiguous
    list(out)
  }
  graph_desc_add(
    p_concatenate,
    args = dots,
    params = list(dimension = dimension),
    infer_fn = infer_fn
  )[[1L]]
}

p_static_slice <- AnvilPrimitive("static_slice")
#' @title Primitive Static Slice
#' @description
#' Extracts a slice from a tensor using static (compile-time) indices.
#' For dynamic indices, use [nvl_dynamic_slice()].
#' @template param_operand
#' @param start_indices (`integer()`)\cr
#'   Start indices (1-based).
#' @param limit_indices (`integer()`)\cr
#'   End indices (exclusive).
#' @param strides (`integer()`)\cr
#'   Step sizes.
#' @return [`tensorish`]
#' @export
nvl_static_slice <- function(operand, start_indices, limit_indices, strides) {
  infer_fn <- function(operand, start_indices, limit_indices, strides) {
    start_attr <- r_to_constant(start_indices - 1L, dtype = "i64", shape = length(start_indices))
    limit_attr <- r_to_constant(limit_indices, dtype = "i64", shape = length(limit_indices))
    strides_attr <- r_to_constant(strides, dtype = "i64", shape = length(strides))
    out <- stablehlo::infer_types_slice(at2vt(operand), start_attr, limit_attr, strides_attr)[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  graph_desc_add(
    p_static_slice,
    args = list(
      operand = operand
    ),
    params = list(
      start_indices = start_indices,
      limit_indices = limit_indices,
      strides = strides
    ),
    infer_fn = infer_fn
  )[[1L]]
}

p_dynamic_slice <- AnvilPrimitive("dynamic_slice")
#' @title Primitive Dynamic Slice
#' @description
#' Extracts a dynamically positioned slice from a tensor.
#' The start position is specified at runtime via tensor indices.
#' @template param_operand
#' @param ... Scalar tensor start indices (1-based), one per dimension.
#' @param slice_sizes (`integer()`)\cr
#'   Size of the slice in each dimension.
#' @section Out Of Bounds Behavior:
#' If the slice would extend beyond the bounds of the operand tensor,
#' the start indices are clamped so that the slice fits within the tensor.
#' This means that out-of-bounds indices will not cause an error, but
#' the effective start position may differ from the requested one.
#'
#' For example, slicing a tensor of shape `c(10)` with `start_indices = 8`
#' and `slice_sizes = 5` will effectively use `start_indices = 6` to keep
#' the slice within bounds.
#' @return [`tensorish`]
#' @export
nvl_dynamic_slice <- function(operand, ..., slice_sizes) {
  start_indices <- list(...)
  infer_fn <- function(operand, ..., slice_sizes) {
    start_indices_avals <- list(...)
    for (i in seq_along(start_indices_avals)) {
      aval <- start_indices_avals[[i]]
      if (length(shape(aval)) != 0L) {
        cli_abort("Start index {i} must be a scalar, but has shape {shape(aval)}")
      }
    }
    out <- AbstractTensor(dtype = operand$dtype, shape = slice_sizes, ambiguous = operand$ambiguous)
    list(out)
  }
  graph_desc_add(
    p_dynamic_slice,
    args = c(list(operand = operand), start_indices),
    params = list(slice_sizes = slice_sizes),
    infer_fn = infer_fn
  )[[1L]]
}

p_dynamic_update_slice <- AnvilPrimitive("dynamic_update_slice")
#' @title Primitive Dynamic Update Slice
#' @description
#' Updates a dynamically positioned slice in a tensor.
#' The start position is specified at runtime via tensor indices.
#' @template param_operand
#' @param update ([`tensorish`])\cr
#'   The values to write at the specified position.
#' @param ... ([`tensorish`])\cr
#'   Scalar tensor start indices (1-based), one per dimension.
#' @inheritSection nvl_dynamic_slice Out Of Bounds Behavior
#' @return [`tensorish`]
#' @export
nvl_dynamic_update_slice <- function(operand, update, ...) {
  start_indices <- list(...)
  infer_fn <- function(operand, update, ...) {
    start_indices_avals <- list(...)
    for (i in seq_along(start_indices_avals)) {
      aval <- start_indices_avals[[i]]
      if (length(shape(aval)) != 0L) {
        cli_abort("Start index {i} must be a scalar, but has shape {shape(aval)}")
      }
    }
    out <- AbstractTensor(dtype = operand$dtype, shape = shape(operand), ambiguous = operand$ambiguous)
    list(out)
  }
  graph_desc_add(
    p_dynamic_update_slice,
    args = c(list(operand = operand, update = update), start_indices),
    params = list(),
    infer_fn = infer_fn
  )[[1L]]
}

# reduction operators

make_reduce_op <- function(prim, infer_fn = infer_reduce) {
  function(operand, dims, drop = TRUE) {
    graph_desc_add(
      prim,
      list(operand = operand),
      params = list(dims = dims, drop = drop),
      infer_fn = infer_fn
    )[[1L]]
  }
}

p_reduce_sum <- AnvilPrimitive("reduce_sum")
#' @title Primitive Sum Reduction
#' @description
#' Sums tensor elements along dimensions.
#' @template param_operand
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop reduced dimensions.
#' @return [`tensorish`]
#' @export
nvl_reduce_sum <- make_reduce_op(p_reduce_sum)

p_reduce_prod <- AnvilPrimitive("reduce_prod")
#' @title Primitive Product Reduction
#' @description
#' Multiplies tensor elements along dimensions.
#' @template param_operand
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop reduced dimensions.
#' @return [`tensorish`]
#' @export
nvl_reduce_prod <- make_reduce_op(p_reduce_prod)

p_reduce_max <- AnvilPrimitive("reduce_max")
#' @title Primitive Max Reduction
#' @description
#' Finds maximum along dimensions.
#' @template param_operand
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop reduced dimensions.
#' @return [`tensorish`]
#' @export
nvl_reduce_max <- make_reduce_op(p_reduce_max)

p_reduce_min <- AnvilPrimitive("reduce_min")
#' @title Primitive Min Reduction
#' @description
#' Finds minimum along dimensions.
#' @template param_operand
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop reduced dimensions.
#' @return [`tensorish`]
#' @export
nvl_reduce_min <- make_reduce_op(p_reduce_min)

p_reduce_any <- AnvilPrimitive("reduce_any")
#' @title Primitive Any Reduction
#' @description
#' Logical OR along dimensions.
#' @template param_operand
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop reduced dimensions.
#' @return [`tensorish`]
#' @export
nvl_reduce_any <- make_reduce_op(p_reduce_any, infer_reduce_boolean)

p_reduce_all <- AnvilPrimitive("reduce_all")
#' @title Primitive All Reduction
#' @description
#' Logical AND along dimensions.
#' @template param_operand
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop reduced dimensions.
#' @return [`tensorish`]
#' @export
nvl_reduce_all <- make_reduce_op(p_reduce_all, infer_reduce_boolean)

# comparison primitives --------------------------------------------------------

infer_compare <- function(lhs, rhs, comparison_direction) {
  check_dtype <- as.character(dtype(lhs))
  compare_type <- if ((check_dtype == "i1") || grepl("^ui", check_dtype)) {
    "UNSIGNED"
  } else if (grepl("^i", check_dtype)) {
    "SIGNED"
  } else {
    "FLOAT"
  }
  out <- stablehlo::infer_types_compare(at2vt(lhs), at2vt(rhs), comparison_direction, compare_type)[[1L]]
  out <- vt2at(out)
  out$ambiguous <- lhs$ambiguous && rhs$ambiguous
  list(out)
}

make_compare_op <- function(prim, direction) {
  infer_fn <- function(lhs, rhs) infer_compare(lhs, rhs, direction)
  function(lhs, rhs) {
    graph_desc_add(prim, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
  }
}

p_eq <- AnvilPrimitive("equal")
#' @title Primitive Equal
#' @description
#' Element-wise equality comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_eq <- make_compare_op(p_eq, "EQ")

p_ne <- AnvilPrimitive("not_equal")
#' @title Primitive Not Equal
#' @description
#' Element-wise inequality comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_ne <- make_compare_op(p_ne, "NE")

p_gt <- AnvilPrimitive("greater")
#' @title Primitive Greater Than
#' @description
#' Element-wise greater than comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_gt <- make_compare_op(p_gt, "GT")

p_ge <- AnvilPrimitive("greater_equal")
#' @title Primitive Greater Equal
#' @description
#' Element-wise greater than or equal comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_ge <- make_compare_op(p_ge, "GE")

p_lt <- AnvilPrimitive("less")
#' @title Primitive Less Than
#' @description
#' Element-wise less than comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_lt <- make_compare_op(p_lt, "LT")

p_le <- AnvilPrimitive("less_equal")
#' @title Primitive Less Equal
#' @description
#' Element-wise less than or equal comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_le <- make_compare_op(p_le, "LE")

# additional simple binary primitives -----------------------------------------

p_max <- AnvilPrimitive("maximum")
#' @title Primitive Maximum
#' @description
#' Element-wise maximum of two tensors.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_max <- make_binary_op(p_max, stablehlo::infer_types_maximum)

p_min <- AnvilPrimitive("minimum")
#' @title Primitive Minimum
#' @description
#' Element-wise minimum of two tensors.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_min <- make_binary_op(p_min, stablehlo::infer_types_minimum)

p_remainder <- AnvilPrimitive("remainder")
#' @title Primitive Remainder
#' @description
#' Element-wise remainder of division.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_remainder <- make_binary_op(p_remainder, stablehlo::infer_types_remainder)

p_and <- AnvilPrimitive("and")
#' @title Primitive And
#' @description
#' Element-wise logical AND.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_and <- make_binary_op(p_and, stablehlo::infer_types_and)

p_not <- AnvilPrimitive("not")
#' @title Primitive Not
#' @description
#' Element-wise logical NOT.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_not <- make_unary_op(p_not, stablehlo::infer_types_not)

p_or <- AnvilPrimitive("or")
#' @title Primitive Or
#' @description
#' Element-wise logical OR.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_or <- make_binary_op(p_or, stablehlo::infer_types_or)

p_xor <- AnvilPrimitive("xor")
#' @title Primitive Xor
#' @description
#' Element-wise logical XOR.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_xor <- make_binary_op(p_xor, stablehlo::infer_types_xor)

infer_shift <- function(lhs, rhs, shift_fn) {
  both_ambiguous <- lhs$ambiguous && rhs$ambiguous
  out <- shift_fn(at2vt(lhs), at2vt(rhs))[[1L]]
  out <- vt2at(out)
  out$ambiguous <- both_ambiguous
  list(out)
}

p_shift_left <- AnvilPrimitive("shift_left")
#' @title Primitive Shift Left
#' @description
#' Element-wise left bit shift.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_shift_left <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_left)
  graph_desc_add(p_shift_left, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
}

p_shift_right_logical <- AnvilPrimitive("shift_right_logical")
#' @title Primitive Logical Shift Right
#' @description
#' Element-wise logical right bit shift.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_shift_right_logical <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_right_logical)
  graph_desc_add(p_shift_right_logical, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
}

p_shift_right_arithmetic <- AnvilPrimitive("shift_right_arithmetic")
#' @title Primitive Arithmetic Shift Right
#' @description
#' Element-wise arithmetic right bit shift.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_shift_right_arithmetic <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_right_arithmetic)
  graph_desc_add(p_shift_right_arithmetic, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
}

p_atan2 <- AnvilPrimitive("atan2")
#' @title Primitive Atan2
#' @description
#' Element-wise atan2 operation.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_atan2 <- make_binary_op(p_atan2, stablehlo::infer_types_atan2)

p_bitcast_convert <- AnvilPrimitive("bitcast_convert")
#' @title Primitive Bitcast Convert
#' @description
#' Reinterprets tensor bits as a different dtype.
#' @template param_operand
#' @template param_dtype
#' @return [`tensorish`]
#' @export
nvl_bitcast_convert <- function(operand, dtype) {
  infer_fn <- function(operand, dtype) {
    lapply(stablehlo::infer_types_bitcast_convert(at2vt(operand), dtype), vt2at)
  }
  graph_desc_add(p_bitcast_convert, list(operand = operand), params = list(dtype = dtype), infer_fn = infer_fn)[[1L]]
}

# unary math primitives ---------------------------------------------------------

p_abs <- AnvilPrimitive("abs")
#' @title Primitive Absolute Value
#' @description
#' Element-wise absolute value.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_abs <- make_unary_op(p_abs, stablehlo::infer_types_abs)

p_sqrt <- AnvilPrimitive("sqrt")
#' @title Primitive Square Root
#' @description
#' Element-wise square root.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_sqrt <- make_unary_op(p_sqrt, stablehlo::infer_types_sqrt)

p_rsqrt <- AnvilPrimitive("rsqrt")

#' @title Primitive Reciprocal Square Root
#' @description
#' Element-wise reciprocal square root.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_rsqrt <- make_unary_op(p_rsqrt, stablehlo::infer_types_rsqrt)

p_log <- AnvilPrimitive("log")
#' @title Primitive Logarithm
#' @description
#' Element-wise natural logarithm.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_log <- make_unary_op(p_log, stablehlo::infer_types_log)

p_tanh <- AnvilPrimitive("tanh")
#' @title Primitive Hyperbolic Tangent
#' @description
#' Element-wise hyperbolic tangent.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_tanh <- make_unary_op(p_tanh, stablehlo::infer_types_tanh)

p_tan <- AnvilPrimitive("tan")
#' @title Primitive Tangent
#' @description
#' Element-wise tangent.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_tan <- make_unary_op(p_tan, stablehlo::infer_types_tan)

p_sine <- AnvilPrimitive("sine")
#' @title Primitive Sine
#' @description
#' Element-wise sine.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_sine <- make_unary_op(p_sine, stablehlo::infer_types_sine)

p_cosine <- AnvilPrimitive("cosine")
#' @title Primitive Cosine
#' @description
#' Element-wise cosine.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_cosine <- make_unary_op(p_cosine, stablehlo::infer_types_cosine)

p_floor <- AnvilPrimitive("floor")
#' @title Primitive Floor
#' @description
#' Element-wise floor.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_floor <- make_unary_op(p_floor, stablehlo::infer_types_floor)

p_ceil <- AnvilPrimitive("ceil")
#' @title Primitive Ceiling
#' @description
#' Element-wise ceiling.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_ceil <- make_unary_op(p_ceil, stablehlo::infer_types_ceil)

p_sign <- AnvilPrimitive("sign")
#' @title Primitive Sign
#' @description
#' Element-wise sign.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_sign <- make_unary_op(p_sign, stablehlo::infer_types_sign)

p_exp <- AnvilPrimitive("exp")
#' @title Primitive Exponential
#' @description
#' Element-wise exponential.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_exp <- make_unary_op(p_exp, stablehlo::infer_types_exponential)

p_expm1 <- AnvilPrimitive("expm1")
#' @title Primitive Exponential Minus One
#' @description
#' Element-wise exp(x) - 1, more accurate for small x.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_expm1 <- make_unary_op(p_expm1, stablehlo::infer_types_exponential_minus_one)

p_log1p <- AnvilPrimitive("log1p")
#' @title Primitive Log Plus One
#' @description
#' Element-wise log(1 + x), more accurate for small x.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_log1p <- make_unary_op(p_log1p, stablehlo::infer_types_log_plus_one)

p_cbrt <- AnvilPrimitive("cbrt")
#' @title Primitive Cube Root
#' @description
#' Element-wise cube root.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_cbrt <- make_unary_op(p_cbrt, stablehlo::infer_types_cbrt)

p_logistic <- AnvilPrimitive("logistic")
#' @title Primitive Logistic (Sigmoid)
#' @description
#' Element-wise logistic sigmoid: 1 / (1 + exp(-x)).
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_logistic <- make_unary_op(p_logistic, stablehlo::infer_types_logistic)

p_is_finite <- AnvilPrimitive("is_finite")
#' @title Primitive Is Finite
#' @description
#' Element-wise check if values are finite (not Inf, -Inf, or NaN).
#' @template param_operand
#' @return [`tensorish`] of boolean type
#' @export
nvl_is_finite <- function(operand) {
  infer_fn <- function(operand) {
    out <- stablehlo::infer_types_is_finite(at2vt(operand))[[1L]]
    list(vt2at(out))
  }
  graph_desc_add(p_is_finite, list(operand = operand), list(), infer_fn = infer_fn)[[1L]]
}

p_popcnt <- AnvilPrimitive("popcnt")
#' @title Primitive Population Count
#' @description
#' Element-wise population count (number of set bits).
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_popcnt <- function(operand) {
  infer_fn <- function(operand) {
    out <- stablehlo::infer_types_popcnt(at2vt(operand))[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  graph_desc_add(p_popcnt, list(operand = operand), list(), infer_fn = infer_fn)[[1L]]
}

p_clamp <- AnvilPrimitive("clamp")
#' @title Primitive Clamp
#' @description
#' Element-wise clamp: max(min_val, min(operand, max_val)).
#' @param min_val ([`tensorish`])\cr
#'   Minimum value (scalar or same shape as operand).
#' @template param_operand
#' @param max_val ([`tensorish`])\cr
#'   Maximum value (scalar or same shape as operand).
#' @return [`tensorish`]
#' @export
nvl_clamp <- function(min_val, operand, max_val) {
  infer_fn <- function(min_val, operand, max_val) {
    out <- stablehlo::infer_types_clamp(at2vt(min_val), at2vt(operand), at2vt(max_val))[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  graph_desc_add(p_clamp, list(min_val = min_val, operand = operand, max_val = max_val), list(), infer_fn = infer_fn)[[
    1L
  ]]
}

p_reverse <- AnvilPrimitive("reverse")
#' @title Primitive Reverse
#' @description
#' Reverses the order of elements along specified dimensions.
#' @template param_operand
#' @param dims (`integer()`)\cr
#'   Dimensions to reverse (1-indexed).
#' @return [`tensorish`]
#' @export
nvl_reverse <- function(operand, dims) {
  infer_fn <- function(operand, dims) {
    # stablehlo uses 0-based indexing
    dims_attr <- r_to_constant(dims - 1L, dtype = "i64", shape = length(dims))
    out <- stablehlo::infer_types_reverse(at2vt(operand), dimensions = dims_attr)[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  graph_desc_add(p_reverse, list(operand = operand), list(dims = dims), infer_fn = infer_fn)[[1L]]
}

p_iota <- AnvilPrimitive("iota")
#' @title Primitive Iota
#' @description
#' Creates a tensor with values increasing along the specified dimension.
#' @param dim (`integer(1)`)\cr
#'   Dimension along which values increase (1-indexed).
#' @template param_dtype
#' @param shape (`integer()`)\cr
#'   Shape of the output tensor.
#' @return [`tensorish`]
#' @export
nvl_iota <- function(dim, dtype, shape) {
  infer_fn <- function(dim, dtype, shape) {
    # stablehlo uses 0-based indexing, anvil uses 1-based
    # Convert dim to Constant as required by stablehlo
    iota_dim_const <- stablehlo::r_to_constant(
      as.integer(dim - 1L),
      dtype = "i64",
      shape = integer(0)
    )
    out <- stablehlo::infer_types_iota(iota_dimension = iota_dim_const, dtype = dtype, shape = shape)[[1L]]
    list(vt2at(out))
  }
  graph_desc_add(
    p_iota,
    list(),
    list(dim = dim, dtype = dtype, shape = shape),
    infer_fn = infer_fn
  )[[1L]]
}

p_pad <- AnvilPrimitive("pad")
#' @title Primitive Pad
#' @description
#' Pads a tensor with a given padding value.
#' @template param_operand
#' @param padding_value ([`tensorish`])\cr
#'   Scalar value to use for padding.
#' @param edge_padding_low (`integer()`)\cr
#'   Amount of padding to add at the start of each dimension.
#' @param edge_padding_high (`integer()`)\cr
#'   Amount of padding to add at the end of each dimension.
#' @param interior_padding (`integer()`)\cr
#'   Amount of padding to add between elements in each dimension.
#' @return [`tensorish`]
#' @export
nvl_pad <- function(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding) {
  infer_fn <- function(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding) {
    rank <- ndims_abstract(operand)
    low_attr <- r_to_constant(edge_padding_low, dtype = "i64", shape = rank)
    high_attr <- r_to_constant(edge_padding_high, dtype = "i64", shape = rank)
    interior_attr <- r_to_constant(interior_padding, dtype = "i64", shape = rank)
    out <- stablehlo::infer_types_pad(
      at2vt(operand),
      at2vt(padding_value),
      edge_padding_low = low_attr,
      edge_padding_high = high_attr,
      interior_padding = interior_attr
    )[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }

  graph_desc_add(
    p_pad,
    list(operand = operand, padding_value = padding_value),
    list(
      edge_padding_low = edge_padding_low,
      edge_padding_high = edge_padding_high,
      interior_padding = interior_padding
    ),
    infer_fn = infer_fn
  )[[1L]]
}

p_round <- AnvilPrimitive("round")
#' @title Primitive Round
#' @description
#' Element-wise rounding.
#' @template param_operand
#' @param method (`character(1)`)\cr
#'   Rounding method ("nearest_even" or "afz").
#' @return [`tensorish`]
#' @export
nvl_round <- function(operand, method = "nearest_even") {
  if (!(method %in% c("nearest_even", "afz"))) {
    cli_abort("method must be one of: 'nearest_even', 'afz', but is {method}")
  }
  infer_fn <- function(operand, method) {
    # both rounding functions have the same inference, so just pick one:
    stablehlo_infer <- stablehlo::infer_types_round_nearest_even
    out <- stablehlo_infer(at2vt(operand))[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  graph_desc_add(p_round, list(operand = operand), list(method = method), infer_fn = infer_fn)[[1L]]
}

# dtype conversion ----------------------------------------------------------------

p_convert <- AnvilPrimitive("convert")
#' @title Primitive Convert
#' @description
#' Converts tensor to a different dtype.
#' @template param_operand
#' @template param_dtype
#' @template param_ambiguous
#' @return [`tensorish`]
#' @export
nvl_convert <- function(operand, dtype, ambiguous = FALSE) {
  dtype <- as_dtype(dtype)
  infer_fn <- function(operand, dtype, ambiguous) {
    list(AbstractTensor(
      dtype = dtype,
      shape = Shape(shape(operand)),
      ambiguous = ambiguous
    ))
  }
  graph_desc_add(
    p_convert,
    list(operand = operand),
    params = list(dtype = dtype, ambiguous = ambiguous),
    infer_fn = infer_fn
  )[[1L]]
}


p_select <- AnvilPrimitive("select")
#' @title Primitive Select
#' @description
#' Selects elements based on a predicate.
#' @param pred ([`tensorish`])\cr
#'   Boolean predicate tensor.
#' @param true_value ([`tensorish`])\cr
#'   Value when pred is true.
#' @param false_value ([`tensorish`])\cr
#'   Value when pred is false.
#' @return [`tensorish`]
#' @export
nvl_select <- function(pred, true_value, false_value) {
  infer_fn <- function(pred, true_value, false_value) {
    both_ambiguous <- true_value$ambiguous && false_value$ambiguous
    out <- stablehlo::infer_types_select(
      at2vt(pred),
      on_true = at2vt(true_value),
      on_false = at2vt(false_value)
    )[[1L]]
    out <- vt2at(out)
    out$ambiguous <- both_ambiguous
    list(out)
  }
  graph_desc_add(p_select, list(pred = pred, true_value = true_value, false_value = false_value), infer_fn = infer_fn)[[
    1L
  ]]
}

# Higher order primitives -------------------------------------------------------

p_if <- AnvilPrimitive("if", subgraphs = c("true_graph", "false_graph"))
#' @title Primitive If
#' @description
#' Conditional execution of branches.
#' @param pred ([`tensorish`])\cr
#'   Scalar boolean predicate.
#' @param true (`expression`)\cr
#'   Expression for true branch.
#' @param false (`expression`)\cr
#'   Expression for false branch.
#' @return Result of the executed branch.
#' @export
nvl_if <- function(pred, true, false) {
  # delayed promise evaluation can cause the value to be added to the wrong graph descriptor
  force(pred)
  true_expr <- rlang::enquo(true)
  false_expr <- rlang::enquo(false)

  # Build sub-graphs for each branch (no inputs, just capture closed-over values)
  # We need to ensure that constants that are captured in both branches receive the same
  # GraphValue if they capture the same constant

  current_desc <- .current_descriptor(silent = TRUE)

  debug_mode <- is.null(current_desc)
  if (debug_mode) {
    current_desc <- local_descriptor()
  }
  # TODO(split pr)

  desc_true <- local_descriptor()
  true_graph <- trace_fn(function() rlang::eval_tidy(true_expr), list(), desc = desc_true, lit_to_tensor = TRUE)
  desc_false <- local_descriptor()

  # TODO: Apply promotion rules to the outputs of the branches

  for (const in desc_true$constants) {
    get_box_or_register_const(desc_false, const)
  }
  false_graph <- trace_fn(function() rlang::eval_tidy(false_expr), list(), desc = desc_false, lit_to_tensor = TRUE)

  for (const in desc_false$constants) {
    get_box_or_register_const(current_desc, const)
  }

  if (!identical(true_graph$out_tree, false_graph$out_tree)) {
    cli_abort("true and false branches must have the same output structure")
  }

  infer_fn <- function(pred, true_graph, false_graph) {
    # the returned values might have different ambiguity, so we need to handle it
    # an output is ambiguous if it's type is ambiguous in both branches
    lapply(seq_along(true_graph$outputs), function(i) {
      aval_true <- true_graph$outputs[[i]]$aval
      aval_false <- true_graph$outputs[[i]]$aval
      if (aval_true$ambiguous && aval_false$ambiguous) {
        return(aval_true)
      }

      aval_true$ambiguous <- FALSE
      return(aval_true)
    })
  }

  out <- graph_desc_add(
    p_if,
    list(pred = pred),
    params = list(true_graph = true_graph, false_graph = false_graph),
    infer_fn = infer_fn,
    desc = current_desc,
    debug_mode = debug_mode
  )
  unflatten(true_graph$out_tree, out)
}

p_while <- AnvilPrimitive("while", subgraphs = c("cond_graph", "body_graph"))
#' @title Primitive While Loop
#' @description
#' Executes a while loop.
#' @param init (`list()`)\cr
#'   Named list of initial state values.
#' @param cond (`function`)\cr
#'   Condition function returning boolean.
#' @param body (`function`)\cr
#'   Body function returning updated state.
#' @return Final state after loop terminates.
#' @export
nvl_while <- function(init, cond, body) {
  # delayed promise evaluation can cause the value to be added to the wrong graph descriptor
  force(init)
  if (!is.function(body)) {
    cli_abort("body must be a function")
  }
  if (!is.function(cond)) {
    cli_abort("cond must be a function")
  }

  state_names <- names(init)

  if (any(state_names == "")) {
    cli_abort("init must have only named arguments")
  }

  current_desc <- .current_descriptor(silent = TRUE)
  debug_mode <- is.null(current_desc)
  if (debug_mode) {
    current_desc <- local_descriptor()
  }

  desc_cond <- local_descriptor()

  cond_graph <- trace_fn(cond, init, desc = desc_cond, lit_to_tensor = TRUE)

  desc_body <- local_descriptor()

  # ensure that constant ids are the same between cond and body
  # inputs don't matter, because we don't inline the sub-graphs into the parent graph
  for (const in desc_cond$constants) {
    get_box_or_register_const(desc_body, const)
  }
  body_graph <- trace_fn(body, init, desc_body, lit_to_tensor = TRUE)

  if (!identical(cond_graph$in_tree, body_graph$in_tree)) {
    cli_abort("cond and body must have the same input structure")
  }

  if (!identical(body_graph$in_tree, body_graph$out_tree)) {
    cli_abort("body must have the same input and output structure")
  }

  # now we register the constants of both sub-graphs (body includes cond's constants) into the graph
  for (const in body_graph$constants) {
    get_box_or_register_const(current_desc, const)
  }

  infer_fn <- function(..., cond_graph, body_graph) {
    outs <- list(...)
    outs_body <- lapply(body_graph$outputs, \(out) out$aval)
    inputs_body <- lapply(body_graph$inputs, \(inp) inp$aval)
    # == ignores ambiguity
    if (!all(sapply(seq_along(outs), \(i) outs[[i]] == outs_body[[i]]))) {
      cli_abort("outs must be have same type as outs_body")
    }
    if (!all(sapply(seq_along(inputs_body), \(i) inputs_body[[i]] == outs_body[[i]]))) {
      cli_abort("inputs_body must be have same type as outs_body")
    }
    # function might change the ambiguity, so we return the body outputs and not the inputs
    return(outs_body)
  }

  out <- graph_desc_add(
    p_while,
    args = lapply(flatten(init), maybe_box_tensorish),
    params = list(cond_graph = cond_graph, body_graph = body_graph),
    infer_fn = infer_fn,
    desc = current_desc,
    debug_mode = debug_mode
  )

  unflatten(body_graph$out_tree, out)
}

# Print primitive
p_print <- AnvilPrimitive("print")
#' @title Primitive Print
#' @description
#' Prints a tensor during execution.
#' Returns the input unchanged.
#' Note: Currently only works on CPU backend.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_print <- function(operand) {
  # HACK: ambiguity is not available in stablehlo, so we need to pre-compute this
  # and pass it as a "param", although it is not really one
  # TODO: We should also include the platform/device, but it is currently not avilable in GraphDescriptor
  dtype_str <- paste0(as.character(dtype(operand)), if (ambiguous_abstract(operand)) "?")
  footer <- sprintf("[ %s{%s} ]", dtype_str, paste0(shape(operand), collapse = ","))
  # slig
  graph_desc_add(p_print, list(operand = operand), list(footer = footer), infer_fn = function(operand, ...) {
    list(operand)
  })[[1L]]
}

# RNG primitives
p_rng_bit_generator <- AnvilPrimitive("rng_bit_generator")
#' @title Primitive RNG Bit Generator
#' @description
#' Generates random bits using the specified algorithm.
#' @param initial_state ([`tensorish`])\cr
#'   RNG state tensor.
#' @param rng_algorithm (`character(1)`)\cr
#'   Algorithm name (default "THREE_FRY").
#' @template param_dtype
#' @param shape (`integer()`)\cr
#'   Output shape.
#' @return List of new state and random tensor.
#' @export
nvl_rng_bit_generator <- function(initial_state, rng_algorithm = "THREE_FRY", dtype, shape) {
  infer_fn <- function(initial_state, rng_algorithm, dtype, shape) {
    lapply(stablehlo::infer_types_rng_bit_generator(at2vt(initial_state), rng_algorithm, dtype, shape), vt2at)
  }
  graph_desc_add(
    p_rng_bit_generator,
    list(initial_state = initial_state),
    params = list(rng_algorithm = rng_algorithm, dtype = dtype, shape = shape),
    infer_fn = infer_fn
  )
}
