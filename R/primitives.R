#' @include utils.R
#' @include type-converters.R
#' @include primitive.R

infer_binary <- function(lhs, rhs) {
  both_ambiguous <- lhs@ambiguous && rhs@ambiguous
  out <- stablehlo::infer_types_generic_biv(st2va(lhs), st2va(rhs))@items[[1L]]
  out <- vt2sa(out)
  out@ambiguous <- both_ambiguous
  list(out)
}

# boolean is i1 -> integerish
infer_binary_integerish <- function(lhs, rhs) {
  both_ambiguous <- lhs@ambiguous && rhs@ambiguous
  out <- stablehlo::infer_types_integerish_biv(st2va(lhs), st2va(rhs))@items[[1L]]
  out <- vt2sa(out)
  out@ambiguous <- both_ambiguous
  list(out)
}

infer_unary <- function(operand) {
  out <- stablehlo::infer_types_generic_uni(st2va(operand))@items[[1L]]
  out <- vt2sa(out)
  out@ambiguous <- operand@ambiguous
  list(out)
}

make_binary_op <- function(prim) {
  function(lhs, rhs) {
    graph_desc_add(prim, list(lhs, rhs), infer_fn = infer_binary)[[1L]]
  }
}

make_binary_integerish_op <- function(prim) {
  function(lhs, rhs) {
    graph_desc_add(prim, list(lhs, rhs), infer_fn = infer_binary_integerish)[[1L]]
  }
}

infer_unary_integerish <- function(operand) {
  out <- stablehlo::infer_types_integerish_uni(st2va(operand))@items[[1L]]
  out <- vt2sa(out)
  out@ambiguous <- operand@ambiguous
  list(out)
}

make_unary_op <- function(prim) {
  function(operand) {
    graph_desc_add(prim, list(operand), infer_fn = infer_unary)[[1L]]
  }
}

make_unary_integerish_op <- function(prim) {
  function(operand) {
    graph_desc_add(prim, list(operand), infer_fn = infer_unary_integerish)[[1L]]
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
    ambiguous = operand@ambiguous
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
    ambiguous = operand@ambiguous
  ))
}


p_fill <- Primitive("fill")
#' @title Primitive Fill
#' @description
#' Creates a tensor filled with a scalar value.
#' @param value (`numeric(1)`)\cr
#'   Scalar value.
#' @template param_shape
#' @template param_dtype
#' @return [`tensorish`]
#' @export
nvl_fill <- function(value, shape, dtype) {
  infer_fill <- function(value, shape, dtype) {
    list(LiteralTensor(data = value, dtype = as_dtype(dtype), shape = shape, ambiguous = FALSE))
  }
  graph_desc_add(
    p_fill,
    list(),
    params = list(value = value, dtype = dtype, shape = shape),
    infer_fn = infer_fill
  )[[1L]]
}

p_add <- Primitive("add")
#' @title Primitive Addition
#' @description
#' Adds two tensors element-wise.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_add <- make_binary_op(p_add)

p_mul <- Primitive("mul")
#' @title Primitive Multiplication
#' @description
#' Multiplies two tensors element-wise.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_mul <- make_binary_op(p_mul)

p_sub <- Primitive("sub")
#' @title Primitive Subtraction
#' @description
#' Subtracts two tensors element-wise.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_sub <- make_binary_op(p_sub)

p_neg <- Primitive("negate")
#' @title Primitive Negation
#' @description
#' Negates a tensor element-wise.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_neg <- make_unary_op(p_neg)

p_div <- Primitive("divide")
#' @title Primitive Division
#' @description
#' Divides two tensors element-wise.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_div <- make_binary_op(p_div)

p_pow <- Primitive("power")
#' @title Primitive Power
#' @description
#' Raises lhs to the power of rhs element-wise.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_pow <- make_binary_op(p_pow)

p_broadcast_in_dim <- Primitive("broadcast_in_dim")
#' @title Primitive Broadcast
#' @description
#' Broadcasts a tensor to a new shape.
#' @template param_operand
#' @param shape_out (`integer()`)\cr
#'   Target shape.
#' @param broadcast_dimensions (`integer()`)\cr
#'   Dimension mapping.
#' @return [`tensorish`]
#' @importFrom stablehlo r_to_constant
#' @export
nvl_broadcast_in_dim <- function(operand, shape_out, broadcast_dimensions) {
  infer_fn <- function(operand, shape_out, broadcast_dimensions) {
    bd_attr <- r_to_constant(
      as.integer(broadcast_dimensions - 1L),
      dtype = "i64",
      shape = length(broadcast_dimensions)
    )
    out <- stablehlo::infer_types_broadcast_in_dim(
      st2va(operand),
      broadcast_dimensions = bd_attr,
      shape_out = shape_out
    )@items[[1L]]
    out <- vt2sa(out)
    out@ambiguous <- operand@ambiguous
    list(out)
  }
  graph_desc_add(
    p_broadcast_in_dim,
    list(operand),
    params = list(
      shape_out = shape_out,
      broadcast_dimensions = broadcast_dimensions
    ),
    infer_fn = infer_fn
  )[[1L]]
}

p_dot_general <- Primitive("dot_general")
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
    out <- stablehlo::infer_types_dot_general(st2va(lhs), st2va(rhs), dot_dimension_numbers = ddn)@items[[1L]]
    list(vt2sa(out))
  }
  graph_desc_add(
    p_dot_general,
    list(lhs, rhs),
    list(contracting_dims = contracting_dims, batching_dims = batching_dims),
    infer_fn = infer_fn
  )[[1L]]
}

p_transpose <- Primitive("transpose")
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
    out <- stablehlo::infer_types_transpose(st2va(operand), permutation = perm_attr)@items[[1L]]
    out <- vt2sa(out)
    out@ambiguous <- operand@ambiguous
    list(out)
  }
  graph_desc_add(
    p_transpose,
    list(operand),
    list(permutation = permutation),
    infer_fn = infer_fn
  )[[1L]]
}

p_reshape <- Primitive("reshape")
#' @title Primitive Reshape
#' @description
#' Reshapes a tensor to a new shape.
#' @template param_operand
#' @template param_shape
#' @return [`tensorish`]
#' @export
nvl_reshape <- function(operand, shape) {
  infer_fn <- function(operand, shape) {
    out <- stablehlo::infer_types_reshape(st2va(operand), shape_out = shape)@items[[1L]]
    out <- vt2sa(out)
    out@ambiguous <- operand@ambiguous
    list(out)
  }
  graph_desc_add(
    p_reshape,
    list(operand),
    params = list(shape = shape),
    infer_fn = infer_fn
  )[[1L]]
}

p_concatenate <- Primitive("concatenate")
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
    all_ambiguous <- all(vapply(operands, \(x) x@ambiguous, logical(1L)))
    vts <- lapply(operands, st2va)
    out <- rlang::exec(stablehlo::infer_types_concatenate, !!!vts, dimension = dimension - 1L)@items[[1L]]
    out <- vt2sa(out)
    out@ambiguous <- all_ambiguous
    list(out)
  }
  graph_desc_add(
    p_concatenate,
    args = dots,
    params = list(dimension = dimension),
    infer_fn = infer_fn
  )[[1L]]
}

p_slice <- Primitive("slice")
#' @title Primitive Slice
#' @description
#' Extracts a slice from a tensor.
#' @template param_operand
#' @param start_indices (`integer()`)\cr
#'   Start indices (1-based).
#' @param limit_indices (`integer()`)\cr
#'   End indices (exclusive).
#' @param strides (`integer()`)\cr
#'   Step sizes.
#' @return [`tensorish`]
#' @export
nvl_slice <- function(operand, start_indices, limit_indices, strides) {
  infer_fn <- function(operand, start_indices, limit_indices, strides) {
    start_attr <- r_to_constant(start_indices - 1L, dtype = "i64", shape = length(start_indices))
    limit_attr <- r_to_constant(limit_indices, dtype = "i64", shape = length(limit_indices))
    strides_attr <- r_to_constant(strides, dtype = "i64", shape = length(strides))
    out <- stablehlo::infer_types_slice(st2va(operand), start_attr, limit_attr, strides_attr)@items[[1L]]
    out <- vt2sa(out)
    out@ambiguous <- operand@ambiguous
    list(out)
  }
  graph_desc_add(
    p_slice,
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

# reduction operators

make_reduce_op <- function(prim, infer_fn = infer_reduce) {
  function(operand, dims, drop = TRUE) {
    graph_desc_add(
      prim,
      list(operand),
      params = list(dims = dims, drop = drop),
      infer_fn = infer_fn
    )[[1L]]
  }
}

p_reduce_sum <- Primitive("reduce_sum")
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

p_reduce_prod <- Primitive("reduce_prod")
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

p_reduce_max <- Primitive("reduce_max")
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

p_reduce_min <- Primitive("reduce_min")
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

p_reduce_any <- Primitive("reduce_any")
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

p_reduce_all <- Primitive("reduce_all")
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
  out <- stablehlo::infer_types_compare(st2va(lhs), st2va(rhs), comparison_direction, "FLOAT")@items[[1L]]
  out <- vt2sa(out)
  out@ambiguous <- lhs@ambiguous && rhs@ambiguous
  list(out)
}

make_compare_op <- function(prim, direction) {
  infer_fn <- function(lhs, rhs) infer_compare(lhs, rhs, direction)
  function(lhs, rhs) {
    graph_desc_add(prim, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
  }
}

p_eq <- Primitive("equal")
#' @title Primitive Equal
#' @description
#' Element-wise equality comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_eq <- make_compare_op(p_eq, "EQ")

p_ne <- Primitive("not_equal")
#' @title Primitive Not Equal
#' @description
#' Element-wise inequality comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_ne <- make_compare_op(p_ne, "NE")

p_gt <- Primitive("greater")
#' @title Primitive Greater Than
#' @description
#' Element-wise greater than comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_gt <- make_compare_op(p_gt, "GT")

p_ge <- Primitive("greater_equal")
#' @title Primitive Greater Equal
#' @description
#' Element-wise greater than or equal comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_ge <- make_compare_op(p_ge, "GE")

p_lt <- Primitive("less")
#' @title Primitive Less Than
#' @description
#' Element-wise less than comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_lt <- make_compare_op(p_lt, "LT")

p_le <- Primitive("less_equal")
#' @title Primitive Less Equal
#' @description
#' Element-wise less than or equal comparison.
#' @template params_lhs_rhs
#' @return [`tensorish`] (boolean)
#' @export
nvl_le <- make_compare_op(p_le, "LE")

# additional simple binary primitives -----------------------------------------

p_max <- Primitive("maximum")
#' @title Primitive Maximum
#' @description
#' Element-wise maximum of two tensors.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_max <- make_binary_op(p_max)

p_min <- Primitive("minimum")
#' @title Primitive Minimum
#' @description
#' Element-wise minimum of two tensors.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_min <- make_binary_op(p_min)

p_remainder <- Primitive("remainder")
#' @title Primitive Remainder
#' @description
#' Element-wise remainder of division.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_remainder <- make_binary_op(p_remainder)

p_and <- Primitive("and")
#' @title Primitive And
#' @description
#' Element-wise logical AND.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_and <- make_binary_integerish_op(p_and)

p_not <- Primitive("not")
#' @title Primitive Not
#' @description
#' Element-wise logical NOT.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_not <- make_unary_integerish_op(p_not)

p_or <- Primitive("or")
#' @title Primitive Or
#' @description
#' Element-wise logical OR.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_or <- make_binary_integerish_op(p_or)

p_xor <- Primitive("xor")
#' @title Primitive Xor
#' @description
#' Element-wise logical XOR.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_xor <- make_binary_integerish_op(p_xor)

infer_shift <- function(lhs, rhs, shift_fn) {
  both_ambiguous <- lhs@ambiguous && rhs@ambiguous
  out <- shift_fn(st2va(lhs), st2va(rhs))@items[[1L]]
  out <- vt2sa(out)
  out@ambiguous <- both_ambiguous
  list(out)
}

p_shift_left <- Primitive("shift_left")
#' @title Primitive Shift Left
#' @description
#' Element-wise left bit shift.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_shift_left <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_left)
  graph_desc_add(p_shift_left, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

p_shift_right_logical <- Primitive("shift_right_logical")
#' @title Primitive Logical Shift Right
#' @description
#' Element-wise logical right bit shift.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_shift_right_logical <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_right_logical)
  graph_desc_add(p_shift_right_logical, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

p_shift_right_arithmetic <- Primitive("shift_right_arithmetic")
#' @title Primitive Arithmetic Shift Right
#' @description
#' Element-wise arithmetic right bit shift.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_shift_right_arithmetic <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_right_arithmetic)
  graph_desc_add(p_shift_right_arithmetic, list(lhs, rhs), infer_fn = infer_fn)[[1L]]
}

p_atan2 <- Primitive("atan2")
#' @title Primitive Atan2
#' @description
#' Element-wise atan2 operation.
#' @template params_lhs_rhs
#' @return [`tensorish`]
#' @export
nvl_atan2 <- make_binary_op(p_atan2)

p_bitcast_convert <- Primitive("bitcast_convert")
#' @title Primitive Bitcast Convert
#' @description
#' Reinterprets tensor bits as a different dtype.
#' @template param_operand
#' @template param_dtype
#' @return [`tensorish`]
#' @export
nvl_bitcast_convert <- function(operand, dtype) {
  infer_fn <- function(operand, dtype) {
    lapply(stablehlo::infer_types_bitcast_convert(st2va(operand), dtype)@items, vt2sa)
  }
  graph_desc_add(p_bitcast_convert, list(operand), params = list(dtype = dtype), infer_fn = infer_fn)[[1L]]
}

# unary math primitives ---------------------------------------------------------

p_abs <- Primitive("abs")
#' @title Primitive Absolute Value
#' @description
#' Element-wise absolute value.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_abs <- make_unary_op(p_abs)

p_sqrt <- Primitive("sqrt")
#' @title Primitive Square Root
#' @description
#' Element-wise square root.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_sqrt <- make_unary_op(p_sqrt)

p_rsqrt <- Primitive("rsqrt")

#' @title Primitive Reciprocal Square Root
#' @description
#' Element-wise reciprocal square root.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_rsqrt <- make_unary_op(p_rsqrt)

p_log <- Primitive("log")
#' @title Primitive Logarithm
#' @description
#' Element-wise natural logarithm.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_log <- make_unary_op(p_log)

p_tanh <- Primitive("tanh")
#' @title Primitive Hyperbolic Tangent
#' @description
#' Element-wise hyperbolic tangent.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_tanh <- make_unary_op(p_tanh)

p_tan <- Primitive("tan")
#' @title Primitive Tangent
#' @description
#' Element-wise tangent.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_tan <- make_unary_op(p_tan)

p_sine <- Primitive("sine")
#' @title Primitive Sine
#' @description
#' Element-wise sine.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_sine <- make_unary_op(p_sine)

p_cosine <- Primitive("cosine")
#' @title Primitive Cosine
#' @description
#' Element-wise cosine.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_cosine <- make_unary_op(p_cosine)

p_floor <- Primitive("floor")
#' @title Primitive Floor
#' @description
#' Element-wise floor.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_floor <- make_unary_op(p_floor)

p_ceil <- Primitive("ceil")
#' @title Primitive Ceiling
#' @description
#' Element-wise ceiling.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_ceil <- make_unary_op(p_ceil)

p_sign <- Primitive("sign")
#' @title Primitive Sign
#' @description
#' Element-wise sign.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_sign <- make_unary_op(p_sign)

p_exp <- Primitive("exp")
#' @title Primitive Exponential
#' @description
#' Element-wise exponential.
#' @template param_operand
#' @return [`tensorish`]
#' @export
nvl_exp <- make_unary_op(p_exp)

p_round <- Primitive("round")
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
    infer_unary(operand)
  }
  graph_desc_add(p_round, list(operand), list(method = method), infer_fn = infer_fn)[[1L]]
}

# dtype conversion ----------------------------------------------------------------

p_convert <- Primitive("convert")
#' @title Primitive Convert
#' @description
#' Converts tensor to a different dtype.
#' @template param_operand
#' @template param_dtype
#' @param ambiguous (`logical(1)`)\cr
#'   Whether the result type is ambiguous.
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
    list(operand),
    params = list(dtype = dtype, ambiguous = ambiguous),
    infer_fn = infer_fn
  )[[1L]]
}


p_select <- Primitive("select")
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
    both_ambiguous <- true_value@ambiguous && false_value@ambiguous
    out <- stablehlo::infer_types_select(
      st2va(pred),
      on_true = st2va(true_value),
      on_false = st2va(false_value)
    )@items[[1L]]
    out <- vt2sa(out)
    out@ambiguous <- both_ambiguous
    list(out)
  }
  graph_desc_add(p_select, list(pred, true_value, false_value), infer_fn = infer_fn)[[1L]]
}

# Higher order primitives -------------------------------------------------------

p_if <- HigherOrderPrimitive("if")
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
  true_expr <- rlang::enquo(true)
  false_expr <- rlang::enquo(false)

  # Build sub-graphs for each branch (no inputs, just capture closed-over values)
  # We need to ensure that constants that are captured in both branches receive the same
  # GraphValue if they capture the same constant

  current_desc <- .current_descriptor(silent = TRUE)

  #debug_mode <- is.null(current_desc)
  #if (debug_mode) {
  #  current_desc <- local_descriptor()
  #}
  # TODO(split pr)

  desc_true <- local_descriptor()
  true_graph <- trace_fn(function() rlang::eval_tidy(true_expr), list(), desc = desc_true)
  desc_false <- local_descriptor()

  # TODO: Apply promotion rules to the outputs of the branches

  for (const in desc_true@constants) {
    get_box_or_register_const(desc_false, const)
  }
  false_graph <- trace_fn(function() rlang::eval_tidy(false_expr), list(), desc = desc_false)

  for (const in desc_false@constants) {
    get_box_or_register_const(current_desc, const)
  }

  if (!identical(true_graph@out_tree, false_graph@out_tree)) {
    cli_abort("true and false branches must have the same output structure")
  }

  infer_fn <- function(pred, true_graph, false_graph) {
    # the returned values might have different ambiguity, so we need to handle it
    # an output is ambiguous if it's type is ambiguous in both branches
    lapply(seq_along(true_graph@outputs), function(i) {
      aval_true <- true_graph@outputs[[i]]@aval
      aval_false <- true_graph@outputs[[i]]@aval
      if (aval_true@ambiguous && aval_false@ambiguous) {
        return(aval_true)
      }

      aval_true@ambiguous <- FALSE
      return(aval_true)
    })
  }

  out <- graph_desc_add(
    p_if,
    list(pred),
    params = list(true_graph = true_graph, false_graph = false_graph),
    infer_fn = infer_fn,
    desc = current_desc #,
    #debug_mode = debug_mode
    # TODO(split pr)
  )
  unflatten(true_graph@out_tree, out)
}

p_while <- HigherOrderPrimitive("while")
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
  #debug_mode <- is.null(current_desc)
  #if (debug_mode) {
  #  current_desc <- local_descriptor()
  #}

  desc_cond <- local_descriptor()

  cond_graph <- trace_fn(cond, init, desc = desc_cond)

  desc_body <- local_descriptor()

  # ensure that constant ids are the same between cond and body
  # inputs don't matter, because we don't inline the sub-graphs into the parent graph
  for (const in desc_cond@constants) {
    get_box_or_register_const(desc_body, const)
  }
  body_graph <- trace_fn(body, init, desc_body)

  if (!identical(cond_graph@in_tree, body_graph@in_tree)) {
    cli_abort("cond and body must have the same input structure")
  }

  if (!identical(body_graph@in_tree, body_graph@out_tree)) {
    cli_abort("body must have the same input and output structure")
  }

  # now we register the constants of both sub-graphs (body includes cond's constants) into the graph
  for (const in body_graph@constants) {
    get_box_or_register_const(current_desc, const)
  }

  infer_fn <- function(..., cond_graph, body_graph) {
    outs <- list(...)
    outs_body <- lapply(body_graph@outputs, \(out) out@aval)
    inputs_body <- lapply(body_graph@inputs, \(inp) inp@aval)
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
    args = flatten(init),
    params = list(cond_graph = cond_graph, body_graph = body_graph),
    infer_fn = infer_fn,
    desc = current_desc #,
    #debug_mode = debug_mode
  )

  unflatten(body_graph@out_tree, out)
}

# RNG primitives
p_rng_bit_generator <- Primitive("rng_bit_generator")
#' @title Primitive RNG Bit Generator
#' @description
#' Generates random bits using the specified algorithm.
#' @param initial_state ([`tensorish`])\cr
#'   RNG state tensor.
#' @param rng_algorithm (`character(1)`)\cr
#'   Algorithm name (default "THREE_FRY").
#' @template param_dtype
#' @param shape_out (`integer()`)\cr
#'   Output shape.
#' @return List of new state and random tensor.
#' @export
nvl_rng_bit_generator <- function(initial_state, rng_algorithm = "THREE_FRY", dtype, shape_out) {
  infer_fn <- function(initial_state, rng_algorithm, dtype, shape_out) {
    lapply(stablehlo::infer_types_rng_bit_generator(st2va(initial_state), rng_algorithm, dtype, shape_out)@items, vt2sa)
  }
  graph_desc_add(
    p_rng_bit_generator,
    list(initial_state),
    params = list(rng_algorithm = rng_algorithm, dtype = dtype, shape_out = shape_out),
    infer_fn = infer_fn
  )
}
