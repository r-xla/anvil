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
    ambiguous = FALSE
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
#' @section Shapes:
#' Output shape is `shape`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_constant()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval(nvl_fill(3.14, shape = c(2, 3), dtype = "f32"))
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
#' For a more user-friendly interface, see [nv_add()], or use the `+` operator.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id add
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_add()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(4, 5, 6))
#'   nvl_add(x, y)
#' })
#' @export
nvl_add <- make_binary_op(p_add, stablehlo::infer_types_add)

p_mul <- AnvilPrimitive("mul")
#' @title Primitive Multiplication
#' @description
#' Multiplies two tensors element-wise.
#' For a more user-friendly interface, see [nv_mul()], or use the `*` operator.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id mul
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_multiply()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(4, 5, 6))
#'   nvl_mul(x, y)
#' })
#' @export
nvl_mul <- make_binary_op(p_mul, stablehlo::infer_types_multiply)

p_sub <- AnvilPrimitive("sub")
#' @title Primitive Subtraction
#' @description
#' Subtracts two tensors element-wise.
#' For a more user-friendly interface, see [nv_sub()], or use the `-` operator.
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id sub
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_subtract()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(4, 5, 6))
#'   nvl_sub(x, y)
#' })
#' @export
nvl_sub <- make_binary_op(p_sub, stablehlo::infer_types_subtract)

p_negate <- AnvilPrimitive("negate")
#' @title Primitive Negation
#' @description
#' Negates a tensor element-wise.
#' Is the same as [nv_negate()]. You can also use the unary `-` operator.
#' @param operand ([`tensorish`])\cr
#'   Tensorish value of data type integer or floating-point.
#' @template return_prim_unary
#' @templateVar primitive_id negate
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_negate()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, -2, 3))
#'   nvl_negate(x)
#' })
#' @export
nvl_negate <- make_unary_op(p_negate, stablehlo::infer_types_negate)

p_div <- AnvilPrimitive("divide")
#' @title Primitive Division
#' @description
#' Divides two tensors element-wise.
#' For a more user-friendly interface, see [nv_div()], or use the `/` operator.
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id divide
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_divide()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(10, 20, 30))
#'   y <- nv_tensor(c(2, 5, 10))
#'   nvl_div(x, y)
#' })
#' @export
nvl_div <- make_binary_op(p_div, stablehlo::infer_types_divide)

p_pow <- AnvilPrimitive("power")
#' @title Primitive Power
#' @description
#' Raises lhs to the power of rhs element-wise.
#' For a more user-friendly interface, see [nv_pow()], or use the `^` operator.
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id power
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_power()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(2, 3, 4))
#'   y <- nv_tensor(c(3, 2, 1))
#'   nvl_pow(x, y)
#' })
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
#' @section Shapes:
#' `length(broadcast_dimensions)` must equal the rank of `operand`.
#' Each dimension of `operand` must either be 1 or match
#' `shape[broadcast_dimensions[i]]`. Output shape is `shape`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_broadcast_in_dim()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   nvl_broadcast_in_dim(x, shape = c(2, 3), broadcast_dimensions = 2L)
#' })
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
#' @section Shapes:
#' Contracting dimensions in `lhs` and `rhs` must have matching sizes.
#' Batching dimensions must also have matching sizes. The output shape
#' is the batching dimensions followed by the remaining
#' (non-contracted, non-batched) dimensions of `lhs`, then `rhs`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_dot_general()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   y <- nv_tensor(matrix(1:6, nrow = 3))
#'   nvl_dot_general(x, y,
#'     contracting_dims = list(2L, 1L),
#'     batching_dims = list(integer(0), integer(0))
#'   )
#' })
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
#' @section Shapes:
#' Output shape is `shape(operand)[permutation]`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_transpose()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   nvl_transpose(x, permutation = c(2L, 1L))
#' })
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
#' @section Shapes:
#' `shape` must have the same number of elements as `operand`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_reshape()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(1:6)
#'   nvl_reshape(x, shape = c(2, 3))
#' })
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
#' For a more user-friendly interface, see [nv_concatenate()], which also handles
#' type promotion and scalar broadcasting.
#' @param ... ([`tensorish`])\cr
#'   Tensors to concatenate. Must all have the same data type and rank.
#'   All dimensions must match except along `dimension`.
#' @param dimension (`integer(1)`)\cr
#'   Dimension along which to concatenate (1-indexed).
#' @return [`tensorish`]\cr
#'   Has the same data type as the inputs.
#'   The output shape matches the inputs in all dimensions except `dimension`,
#'   which is the sum of the input sizes along that dimension.
#'   It is ambiguous if all inputs are ambiguous.
#' @templateVar primitive_id concatenate
#' @template section_rules
#' @section Shapes:
#' All inputs must have the same rank and shape except along `dimension`.
#' The output dimension size along `dimension` is the sum of the input
#' dimension sizes along `dimension`.
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_concatenate()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(4, 5, 6))
#'   nvl_concatenate(x, y, dimension = 1L)
#' })
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
#' @section Shapes:
#' `start_indices`, `limit_indices`, and `strides` must each have
#' length equal to `rank(operand)`. Output shape is
#' `ceiling((limit_indices - start_indices) / strides)`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_slice()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(1:10)
#'   nvl_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 1L)
#' })
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
#' @param ... ([`tensorish`] of integer type)\cr
#'   Scalar start indices (1-based), one per dimension.
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
#' @section Shapes:
#' Each start index in `...` must be a scalar tensor. The number of
#' start indices must equal `rank(operand)`. `slice_sizes` must satisfy
#' `slice_sizes <= shape(operand)` per dimension. Output shape is
#' `slice_sizes`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_dynamic_slice()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(1:10)
#'   start <- nv_scalar(3L)
#'   nvl_dynamic_slice(x, start, slice_sizes = 3L)
#' })
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
#' @param ... ([`tensorish`] of integer type)\cr
#'   Scalar start indices (1-based), one per dimension.
#' @inheritSection nvl_dynamic_slice Out Of Bounds Behavior
#' @return [`tensorish`]
#' @section Shapes:
#' `update` must have the same rank as `operand`, with
#' `shape(update) <= shape(operand)` per dimension. Each start index
#' in `...` must be a scalar tensor. The output has the same shape as
#' `operand`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_dynamic_update_slice()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(1:5)
#'   update <- nv_tensor(c(10L, 20L))
#'   start <- nv_scalar(2L)
#'   nvl_dynamic_update_slice(x, update, start)
#' })
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
#' Sums tensor elements along the specified dimensions.
#' Is the same as [nv_reduce_sum()].
#' @template param_prim_operand_any
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce
#' @templateVar primitive_id reduce_sum
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_reduce()] with [stablehlo::hlo_add()] as the reducer.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   nvl_reduce_sum(x, dims = 1L)
#' })
#' @export
nvl_reduce_sum <- make_reduce_op(p_reduce_sum)

p_reduce_prod <- AnvilPrimitive("reduce_prod")
#' @title Primitive Product Reduction
#' @description
#' Multiplies tensor elements along the specified dimensions.
#' Is the same as [nv_reduce_prod()].
#' @template param_prim_operand_any
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce
#' @templateVar primitive_id reduce_prod
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_reduce()] with [stablehlo::hlo_multiply()] as the reducer.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   nvl_reduce_prod(x, dims = 1L)
#' })
#' @export
nvl_reduce_prod <- make_reduce_op(p_reduce_prod)

p_reduce_max <- AnvilPrimitive("reduce_max")
#' @title Primitive Max Reduction
#' @description
#' Finds the maximum of tensor elements along the specified dimensions.
#' Is the same as [nv_reduce_max()].
#' @template param_prim_operand_any
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce
#' @templateVar primitive_id reduce_max
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_reduce()] with [stablehlo::hlo_maximum()] as the reducer.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   nvl_reduce_max(x, dims = 1L)
#' })
#' @export
nvl_reduce_max <- make_reduce_op(p_reduce_max)

p_reduce_min <- AnvilPrimitive("reduce_min")
#' @title Primitive Min Reduction
#' @description
#' Finds the minimum of tensor elements along the specified dimensions.
#' Is the same as [nv_reduce_min()].
#' @template param_prim_operand_any
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce
#' @templateVar primitive_id reduce_min
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_reduce()] with [stablehlo::hlo_minimum()] as the reducer.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(1:6, nrow = 2))
#'   nvl_reduce_min(x, dims = 1L)
#' })
#' @export
nvl_reduce_min <- make_reduce_op(p_reduce_min)

p_reduce_any <- AnvilPrimitive("reduce_any")
#' @title Primitive Any Reduction
#' @description
#' Performs logical OR along the specified dimensions.
#' Is the same as [nv_reduce_any()].
#' @template param_prim_operand_boolean
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce_boolean
#' @templateVar primitive_id reduce_any
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_reduce()] with [stablehlo::hlo_or()] as the reducer.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2))
#'   nvl_reduce_any(x, dims = 1L)
#' })
#' @export
nvl_reduce_any <- make_reduce_op(p_reduce_any, infer_reduce_boolean)

p_reduce_all <- AnvilPrimitive("reduce_all")
#' @title Primitive All Reduction
#' @description
#' Performs logical AND along the specified dimensions.
#' Is the same as [nv_reduce_all()].
#' @template param_prim_operand_boolean
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce_boolean
#' @templateVar primitive_id reduce_all
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_reduce()] with [stablehlo::hlo_and()] as the reducer.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2))
#'   nvl_reduce_all(x, dims = 1L)
#' })
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
#' For a more user-friendly interface, see [nv_eq()], or use the `==` operator.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "EQ"`.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(1, 3, 2))
#'   nvl_eq(x, y)
#' })
#' @export
nvl_eq <- make_compare_op(p_eq, "EQ")

p_ne <- AnvilPrimitive("not_equal")
#' @title Primitive Not Equal
#' @description
#' Element-wise inequality comparison.
#' For a more user-friendly interface, see [nv_ne()], or use the `!=` operator.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id not_equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "NE"`.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(1, 3, 2))
#'   nvl_ne(x, y)
#' })
#' @export
nvl_ne <- make_compare_op(p_ne, "NE")

p_gt <- AnvilPrimitive("greater")
#' @title Primitive Greater Than
#' @description
#' Element-wise greater than comparison.
#' For a more user-friendly interface, see [nv_gt()], or use the `>` operator.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id greater
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "GT"`.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(3, 2, 1))
#'   nvl_gt(x, y)
#' })
#' @export
nvl_gt <- make_compare_op(p_gt, "GT")

p_ge <- AnvilPrimitive("greater_equal")
#' @title Primitive Greater Equal
#' @description
#' Element-wise greater than or equal comparison.
#' For a more user-friendly interface, see [nv_ge()], or use the `>=` operator.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id greater_equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "GE"`.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(3, 2, 1))
#'   nvl_ge(x, y)
#' })
#' @export
nvl_ge <- make_compare_op(p_ge, "GE")

p_lt <- AnvilPrimitive("less")
#' @title Primitive Less Than
#' @description
#' Element-wise less than comparison.
#' For a more user-friendly interface, see [nv_lt()], or use the `<` operator.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id less
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "LT"`.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(3, 2, 1))
#'   nvl_lt(x, y)
#' })
#' @export
nvl_lt <- make_compare_op(p_lt, "LT")

p_le <- AnvilPrimitive("less_equal")
#' @title Primitive Less Equal
#' @description
#' Element-wise less than or equal comparison.
#' For a more user-friendly interface, see [nv_le()], or use the `<=` operator.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id less_equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "LE"`.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   y <- nv_tensor(c(3, 2, 1))
#'   nvl_le(x, y)
#' })
#' @export
nvl_le <- make_compare_op(p_le, "LE")

# additional simple binary primitives -----------------------------------------

p_max <- AnvilPrimitive("maximum")
#' @title Primitive Maximum
#' @description
#' Element-wise maximum of two tensors.
#' For a more user-friendly interface, see [nv_max()].
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id maximum
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_maximum()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 5, 3))
#'   y <- nv_tensor(c(4, 2, 6))
#'   nvl_max(x, y)
#' })
#' @export
nvl_max <- make_binary_op(p_max, stablehlo::infer_types_maximum)

p_min <- AnvilPrimitive("minimum")
#' @title Primitive Minimum
#' @description
#' Element-wise minimum of two tensors.
#' For a more user-friendly interface, see [nv_min()].
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id minimum
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_minimum()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 5, 3))
#'   y <- nv_tensor(c(4, 2, 6))
#'   nvl_min(x, y)
#' })
#' @export
nvl_min <- make_binary_op(p_min, stablehlo::infer_types_minimum)

p_remainder <- AnvilPrimitive("remainder")
#' @title Primitive Remainder
#' @description
#' Element-wise remainder of division.
#' For a more user-friendly interface, see [nv_remainder()], or use the `%%` operator.
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id remainder
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_remainder()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(7, 10, 15))
#'   y <- nv_tensor(c(3, 4, 6))
#'   nvl_remainder(x, y)
#' })
#' @export
nvl_remainder <- make_binary_op(p_remainder, stablehlo::infer_types_remainder)

p_and <- AnvilPrimitive("and")
#' @title Primitive And
#' @description
#' Element-wise logical AND.
#' For a more user-friendly interface, see [nv_and()], or use the `&` operator.
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id and
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_and()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(TRUE, FALSE, TRUE))
#'   y <- nv_tensor(c(TRUE, TRUE, FALSE))
#'   nvl_and(x, y)
#' })
#' @export
nvl_and <- make_binary_op(p_and, stablehlo::infer_types_and)

p_not <- AnvilPrimitive("not")
#' @title Primitive Not
#' @description
#' Element-wise logical NOT.
#' Is the same as [nv_not()].
#' @param operand ([`tensorish`])\cr
#'   Tensorish value of data type boolean, integer, or unsigned integer.
#' @template return_prim_unary
#' @templateVar primitive_id not
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_not()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(TRUE, FALSE, TRUE))
#'   nvl_not(x)
#' })
#' @export
nvl_not <- make_unary_op(p_not, stablehlo::infer_types_not)

p_or <- AnvilPrimitive("or")
#' @title Primitive Or
#' @description
#' Element-wise logical OR.
#' For a more user-friendly interface, see [nv_or()], or use the `|` operator.
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id or
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_or()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(TRUE, FALSE, TRUE))
#'   y <- nv_tensor(c(TRUE, TRUE, FALSE))
#'   nvl_or(x, y)
#' })
#' @export
nvl_or <- make_binary_op(p_or, stablehlo::infer_types_or)

p_xor <- AnvilPrimitive("xor")
#' @title Primitive Xor
#' @description
#' Element-wise logical XOR.
#' For a more user-friendly interface, see [nv_xor()].
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id xor
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_xor()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(TRUE, FALSE, TRUE))
#'   y <- nv_tensor(c(TRUE, TRUE, FALSE))
#'   nvl_xor(x, y)
#' })
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
#' For a more user-friendly interface, see [nv_shift_left()].
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id shift_left
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_shift_left()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1L, 2L, 4L))
#'   y <- nv_tensor(c(1L, 2L, 1L))
#'   nvl_shift_left(x, y)
#' })
#' @export
nvl_shift_left <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_left)
  graph_desc_add(p_shift_left, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
}

p_shift_right_logical <- AnvilPrimitive("shift_right_logical")
#' @title Primitive Logical Shift Right
#' @description
#' Element-wise logical right bit shift.
#' For a more user-friendly interface, see [nv_shift_right_logical()].
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id shift_right_logical
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_shift_right_logical()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(8L, 16L, 32L))
#'   y <- nv_tensor(c(1L, 2L, 3L))
#'   nvl_shift_right_logical(x, y)
#' })
#' @export
nvl_shift_right_logical <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_right_logical)
  graph_desc_add(p_shift_right_logical, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
}

p_shift_right_arithmetic <- AnvilPrimitive("shift_right_arithmetic")
#' @title Primitive Arithmetic Shift Right
#' @description
#' Element-wise arithmetic right bit shift.
#' For a more user-friendly interface, see [nv_shift_right_arithmetic()].
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id shift_right_arithmetic
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_shift_right_arithmetic()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(8L, -16L, 32L))
#'   y <- nv_tensor(c(1L, 2L, 3L))
#'   nvl_shift_right_arithmetic(x, y)
#' })
#' @export
nvl_shift_right_arithmetic <- function(lhs, rhs) {
  infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_right_arithmetic)
  graph_desc_add(p_shift_right_arithmetic, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
}

p_atan2 <- AnvilPrimitive("atan2")
#' @title Primitive Atan2
#' @description
#' Element-wise atan2 operation.
#' For a more user-friendly interface, see [nv_atan2()].
#' @template params_prim_lhs_rhs_float
#' @template return_prim_binary
#' @templateVar primitive_id atan2
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_atan2()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   y <- nv_tensor(c(1, 0, -1))
#'   x <- nv_tensor(c(0, 1, 0))
#'   nvl_atan2(y, x)
#' })
#' @export
nvl_atan2 <- make_binary_op(p_atan2, stablehlo::infer_types_atan2)

p_bitcast_convert <- AnvilPrimitive("bitcast_convert")
#' @title Primitive Bitcast Convert
#' @description
#' Reinterprets tensor bits as a different dtype.
#' @template param_operand
#' @template param_dtype
#' @return [`tensorish`]
#' @section Shapes:
#' If the source and target types have the same bit width, the output
#' has the same shape as `operand`. Otherwise the last dimension is
#' adjusted based on the bit-width ratio.
#' @section StableHLO:
#' Calls [stablehlo::hlo_bitcast_convert()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(1L)
#'   nvl_bitcast_convert(x, dtype = "f32")
#' })
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
#' Is the same as [nv_abs()]. You can also use [abs()].
#' @template param_prim_operand_signed_numeric
#' @template return_prim_unary
#' @templateVar primitive_id abs
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_abs()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(-1, 2, -3))
#'   nvl_abs(x)
#' })
#' @export
nvl_abs <- make_unary_op(p_abs, stablehlo::infer_types_abs)

p_sqrt <- AnvilPrimitive("sqrt")
#' @title Primitive Square Root
#' @description
#' Element-wise square root.
#' Is the same as [nv_sqrt()]. You can also use [sqrt()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id sqrt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_sqrt()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 4, 9))
#'   nvl_sqrt(x)
#' })
#' @export
nvl_sqrt <- make_unary_op(p_sqrt, stablehlo::infer_types_sqrt)

p_rsqrt <- AnvilPrimitive("rsqrt")
#' @title Primitive Reciprocal Square Root
#' @description
#' Element-wise reciprocal square root.
#' Is the same as [nv_rsqrt()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id rsqrt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_rsqrt()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 4, 9))
#'   nvl_rsqrt(x)
#' })
#' @export
nvl_rsqrt <- make_unary_op(p_rsqrt, stablehlo::infer_types_rsqrt)

p_log <- AnvilPrimitive("log")
#' @title Primitive Logarithm
#' @description
#' Element-wise natural logarithm.
#' Is the same as [nv_log()]. You can also use [log()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id log
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_log()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2.718, 7.389))
#'   nvl_log(x)
#' })
#' @export
nvl_log <- make_unary_op(p_log, stablehlo::infer_types_log)

p_tanh <- AnvilPrimitive("tanh")
#' @title Primitive Hyperbolic Tangent
#' @description
#' Element-wise hyperbolic tangent.
#' Is the same as [nv_tanh()]. You can also use [tanh()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id tanh
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_tanh()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(-1, 0, 1))
#'   nvl_tanh(x)
#' })
#' @export
nvl_tanh <- make_unary_op(p_tanh, stablehlo::infer_types_tanh)

p_tan <- AnvilPrimitive("tan")
#' @title Primitive Tangent
#' @description
#' Element-wise tangent.
#' Is the same as [nv_tan()]. You can also use [tan()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id tan
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_tan()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, 0.5, 1))
#'   nvl_tan(x)
#' })
#' @export
nvl_tan <- make_unary_op(p_tan, stablehlo::infer_types_tan)

p_sine <- AnvilPrimitive("sine")
#' @title Primitive Sine
#' @description
#' Element-wise sine.
#' Is the same as [nv_sine()]. You can also use [sin()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id sine
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_sine()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, pi / 2, pi))
#'   nvl_sine(x)
#' })
#' @export
nvl_sine <- make_unary_op(p_sine, stablehlo::infer_types_sine)

p_cosine <- AnvilPrimitive("cosine")
#' @title Primitive Cosine
#' @description
#' Element-wise cosine.
#' Is the same as [nv_cosine()]. You can also use [cos()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id cosine
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_cosine()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, pi / 2, pi))
#'   nvl_cosine(x)
#' })
#' @export
nvl_cosine <- make_unary_op(p_cosine, stablehlo::infer_types_cosine)

p_floor <- AnvilPrimitive("floor")
#' @title Primitive Floor
#' @description
#' Element-wise floor.
#' Is the same as [nv_floor()]. You can also use [floor()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id floor
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_floor()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1.2, 2.7, -1.5))
#'   nvl_floor(x)
#' })
#' @export
nvl_floor <- make_unary_op(p_floor, stablehlo::infer_types_floor)

p_ceil <- AnvilPrimitive("ceil")
#' @title Primitive Ceiling
#' @description
#' Element-wise ceiling.
#' Is the same as [nv_ceil()]. You can also use [ceiling()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id ceil
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_ceil()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1.2, 2.7, -1.5))
#'   nvl_ceil(x)
#' })
#' @export
nvl_ceil <- make_unary_op(p_ceil, stablehlo::infer_types_ceil)

p_sign <- AnvilPrimitive("sign")
#' @title Primitive Sign
#' @description
#' Element-wise sign.
#' Is the same as [nv_sign()]. You can also use [sign()].
#' @template param_prim_operand_signed_numeric
#' @template return_prim_unary
#' @templateVar primitive_id sign
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_sign()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(-3, 0, 5))
#'   nvl_sign(x)
#' })
#' @export
nvl_sign <- make_unary_op(p_sign, stablehlo::infer_types_sign)

p_exp <- AnvilPrimitive("exp")
#' @title Primitive Exponential
#' @description
#' Element-wise exponential.
#' Is the same as [nv_exp()]. You can also use [exp()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id exp
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_exponential()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, 1, 2))
#'   nvl_exp(x)
#' })
#' @export
nvl_exp <- make_unary_op(p_exp, stablehlo::infer_types_exponential)

p_expm1 <- AnvilPrimitive("expm1")
#' @title Primitive Exponential Minus One
#' @description
#' Element-wise exp(x) - 1, more accurate for small x.
#' Is the same as [nv_expm1()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id expm1
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_exponential_minus_one()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, 0.001, 1))
#'   nvl_expm1(x)
#' })
#' @export
nvl_expm1 <- make_unary_op(p_expm1, stablehlo::infer_types_exponential_minus_one)

p_log1p <- AnvilPrimitive("log1p")
#' @title Primitive Log Plus One
#' @description
#' Element-wise log(1 + x), more accurate for small x.
#' Is the same as [nv_log1p()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id log1p
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_log_plus_one()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(0, 0.001, 1))
#'   nvl_log1p(x)
#' })
#' @export
nvl_log1p <- make_unary_op(p_log1p, stablehlo::infer_types_log_plus_one)

p_cbrt <- AnvilPrimitive("cbrt")
#' @title Primitive Cube Root
#' @description
#' Element-wise cube root.
#' Is the same as [nv_cbrt()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id cbrt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_cbrt()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 8, 27))
#'   nvl_cbrt(x)
#' })
#' @export
nvl_cbrt <- make_unary_op(p_cbrt, stablehlo::infer_types_cbrt)

p_logistic <- AnvilPrimitive("logistic")
#' @title Primitive Logistic (Sigmoid)
#' @description
#' Element-wise logistic sigmoid: 1 / (1 + exp(-x)).
#' Is the same as [nv_logistic()].
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id logistic
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_logistic()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(-2, 0, 2))
#'   nvl_logistic(x)
#' })
#' @export
nvl_logistic <- make_unary_op(p_logistic, stablehlo::infer_types_logistic)

p_is_finite <- AnvilPrimitive("is_finite")
#' @title Primitive Is Finite
#' @description
#' Element-wise check if values are finite (not Inf, -Inf, or NaN).
#' Is the same as [nv_is_finite()].
#' @template param_prim_operand_float
#' @return [`tensorish`]\cr
#'   Has the same shape as the input and boolean data type.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id is_finite
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_is_finite()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, Inf, NaN, -Inf, 0))
#'   nvl_is_finite(x)
#' })
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
#' Is the same as [nv_popcnt()].
#' @param operand ([`tensorish`])\cr
#'   Tensorish value of data type integer or unsigned integer.
#' @template return_prim_unary
#' @templateVar primitive_id popcnt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_popcnt()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(7L, 3L, 15L))
#'   nvl_popcnt(x)
#' })
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
#' @section Shapes:
#' `min_val` and `max_val` must each be either scalar or the same
#' shape as `operand`. The output has the same shape as `operand`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_clamp()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(-1, 0.5, 2))
#'   nvl_clamp(nv_scalar(0), x, nv_scalar(1))
#' })
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
#' @section Shapes:
#' Output has the same shape as `operand`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_reverse()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3, 4, 5))
#'   nvl_reverse(x, dims = 1L)
#' })
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
#' @param start (`integer(1)`)\cr
#'   Starting value.
#' @template param_ambiguous
#' @return [`tensorish`]
#' @section Shapes:
#' Output shape is `shape`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_iota()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval(nvl_iota(dim = 1L, dtype = "i32", shape = 5L))
#' @export
nvl_iota <- function(dim, dtype, shape, start = 1L, ambiguous = FALSE) {
  infer_fn <- function(dim, dtype, shape, start, ambiguous) {
    # stablehlo uses 0-based indexing, anvil uses 1-based
    # Convert dim to Constant as required by stablehlo
    iota_dim_const <- stablehlo::r_to_constant(
      as.integer(dim - 1L),
      dtype = "i64",
      shape = integer(0)
    )
    # Just for the checks
    stablehlo::infer_types_iota(iota_dimension = iota_dim_const, dtype = dtype, shape = shape)[[1L]]

    list(IotaTensor(shape = shape, dtype = dtype, dimension = dim, start = start, ambiguous = ambiguous))
  }
  result <- graph_desc_add(
    p_iota,
    list(),
    list(dim = dim, dtype = dtype, shape = shape, start = start, ambiguous = ambiguous),
    infer_fn = infer_fn
  )[[1L]]

  result
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
#' @section Shapes:
#' `padding_value` must be scalar. `edge_padding_low`,
#' `edge_padding_high`, and `interior_padding` must each have length
#' equal to `rank(operand)`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_pad()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   nvl_pad(x, nv_scalar(0),
#'     edge_padding_low = 2L, edge_padding_high = 1L, interior_padding = 0L
#'   )
#' })
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
#' @param operand ([`tensorish`] of floating-point type)\cr
#'   Operand.
#' @param method (`character(1)`)\cr
#'   Rounding method ("nearest_even" or "afz").
#' @return [`tensorish`]
#' @section Shapes:
#' Operand can have any shape. The output has the same shape.
#' @section StableHLO:
#' Calls [stablehlo::hlo_round_nearest_even()] or
#' [stablehlo::hlo_round_nearest_afz()] depending on the `method` parameter.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1.4, 2.5, 3.6))
#'   nvl_round(x)
#' })
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
#' @section Shapes:
#' Output has the same shape as `operand`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_convert()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1L, 2L, 3L))
#'   nvl_convert(x, dtype = "f32")
#' })
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
#' @title Primitive Ifelse
#' @description
#' Selects elements based on a predicate.
#' @param pred ([`tensorish`] of boolean type)\cr
#'   Predicate tensor.
#' @param true_value ([`tensorish`])\cr
#'   Value when pred is true.
#' @param false_value ([`tensorish`])\cr
#'   Value when pred is false.
#' @return [`tensorish`]
#' @section Shapes:
#' `pred` must be either scalar or the same shape as `true_value`.
#' `true_value` and `false_value` must have the same shape. Output has
#' the shape of `true_value`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_select()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   pred <- nv_tensor(c(TRUE, FALSE, TRUE))
#'   nvl_ifelse(pred, nv_tensor(c(1, 2, 3)), nv_tensor(c(4, 5, 6)))
#' })
#' @export
nvl_ifelse <- function(pred, true_value, false_value) {
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
#' @param pred ([`tensorish`] of boolean type, scalar)\cr
#'   Predicate.
#' @param true (`expression`)\cr
#'   Expression for true branch.
#' @param false (`expression`)\cr
#'   Expression for false branch.
#' @return Result of the executed branch.
#' @section Shapes:
#' `pred` must be scalar. Both branches must return outputs with the same shapes.
#' @section StableHLO:
#' Calls [stablehlo::hlo_if()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval(nvl_if(nv_scalar(TRUE), nv_scalar(1), nv_scalar(2)))
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

  desc_true <- local_descriptor()
  true_graph <- trace_fn(function() rlang::eval_tidy(true_expr), list(), desc = desc_true, lit_to_tensor = TRUE)
  desc_false <- local_descriptor()

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

  # TODO: Apply promotion rules to the outputs of the branches

  infer_fn <- function(pred, true_graph, false_graph) {
    # The returned values might have different ambiguity, so we need to handle it.
    # An output is ambiguous if its type is ambiguous in both branches.
    lapply(seq_along(true_graph$outputs), function(i) {
      aval_true <- true_graph$outputs[[i]]$aval
      aval_false <- false_graph$outputs[[i]]$aval
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
#' @section Shapes:
#' `cond` must return a scalar boolean. `body` must return the same shapes as `init`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_while()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   nvl_while(
#'     init = list(i = nv_scalar(0L), total = nv_scalar(0L)),
#'     cond = function(i, total) nvl_lt(i, nv_scalar(5L)),
#'     body = function(i, total) list(
#'       i = nvl_add(i, nv_scalar(1L)),
#'       total = nvl_add(total, i)
#'     )
#'   )
#' })
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
    # ignore ambiguity when comparing dtypes
    if (!all(sapply(seq_along(outs), \(i) eq_type(outs[[i]], outs_body[[i]], ambiguity = FALSE)))) {
      cli_abort("outs must be have same type as outs_body")
    }
    if (!all(sapply(seq_along(inputs_body), \(i) eq_type(inputs_body[[i]], outs_body[[i]], ambiguity = FALSE)))) {
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
#' @section Shapes:
#' Output has the same shape as `operand`.
#' @section StableHLO:
#' Uses [stablehlo::hlo_custom_call()] internally.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3))
#'   nvl_print(x)
#' })
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
#' @param initial_state (1-d [`tensorish`] of type `ui64`)\cr
#'   RNG state tensor.
#' @param rng_algorithm (`character(1)`)\cr
#'   Algorithm name (default "THREE_FRY").
#' @template param_dtype
#' @param shape (`integer()`)\cr
#'   Output shape.
#' @return List of new state and random tensor.
#' @section Shapes:
#' `initial_state` must be 1-d. Returns a list with the updated state
#' (same shape as `initial_state`) and a random tensor with the
#' specified `shape`.
#' @section StableHLO:
#' Calls [stablehlo::hlo_rng_bit_generator()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   state <- nv_tensor(c(0L, 0L), dtype = "ui64")
#'   nvl_rng_bit_generator(state, dtype = "f32", shape = c(3))
#' })
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

p_scatter <- AnvilPrimitive("scatter", subgraphs = "update_computation_graph")
#' @title Primitive Scatter
#' @description
#' Produces a result tensor equal to the input tensor except that
#' slices specified by scatter_indices are updated with values from the update tensor.
#' @param input ([`tensorish`])\cr
#'   Input tensor to scatter into.
#' @param scatter_indices ([`tensorish`] of integer type)\cr
#'   Indices tensor.
#' @param update ([`tensorish`])\cr
#'   Update values tensor.
#' @param update_window_dims (`integer()`)\cr
#'   Update window dimensions.
#' @param inserted_window_dims (`integer()`)\cr
#'   Inserted window dimensions.
#' @param input_batching_dims (`integer()`)\cr
#'   Input batching dimensions.
#' @param scatter_indices_batching_dims (`integer()`)\cr
#'   Scatter indices batching dimensions.
#' @param scatter_dims_to_operand_dims (`integer()`)\cr
#'   Mapping from scatter indices to operand dimensions.
#' @param index_vector_dim (`integer(1)`)\cr
#'   Dimension in scatter_indices containing the index vectors.
#' @param indices_are_sorted (`logical(1)`)\cr
#'   Whether indices are sorted.
#' @param unique_indices (`logical(1)`)\cr
#'   Whether indices are unique.
#' @param update_computation (`function`)\cr
#'   Binary function to combine existing and update values.
#' @return [`tensorish`]
#' @section Shapes:
#' Output has the same shape as `input`. See
#' [stablehlo::hlo_scatter()] for detailed dimension constraints on
#' `scatter_indices`, `update`, and the dimension mapping parameters.
#' @section StableHLO:
#' Calls [stablehlo::hlo_scatter()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   input <- nv_tensor(c(0, 0, 0, 0, 0))
#'   indices <- nv_tensor(matrix(c(1L, 3L), ncol = 1))
#'   updates <- nv_tensor(c(10, 30))
#'   nvl_scatter(
#'     input, indices, updates,
#'     update_window_dims = integer(0),
#'     inserted_window_dims = 1L,
#'     input_batching_dims = integer(0),
#'     scatter_indices_batching_dims = integer(0),
#'     scatter_dims_to_operand_dims = 1L,
#'     index_vector_dim = 2L
#'   )
#' })
#' @export
nvl_scatter <- function(
  input,
  scatter_indices,
  update,
  update_window_dims,
  inserted_window_dims,
  input_batching_dims,
  scatter_indices_batching_dims,
  scatter_dims_to_operand_dims,
  index_vector_dim,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  update_computation = NULL
) {
  # otherwise, delayed promise evaluation means they might be added to the update_descriptor
  force(input)
  force(scatter_indices)
  force(update)
  if (is.null(update_computation)) {
    update_computation <- function(old, new) new
  } else if (!is.function(update_computation)) {
    cli_abort("update_computation must be a function")
  }

  current_desc <- .current_descriptor(silent = TRUE)
  debug_mode <- is.null(current_desc)
  if (debug_mode) {
    current_desc <- local_descriptor()
  }

  # Trace the update computation function
  # For scatter, the update computation takes 2 scalar arguments (current, update)
  desc_update <- local_descriptor()

  # Create dummy arguments for tracing - use the input's dtype
  input_dtype <- dtype_abstract(input)
  update_dtype <- dtype_abstract(update)
  if (input_dtype != update_dtype) {
    cli_abort("input and update must have the same dtype")
  }

  dummy_args <- list(
    AbstractTensor(dtype = input_dtype, shape = Shape(integer()), ambiguous = ambiguous_abstract(input)),
    AbstractTensor(dtype = input_dtype, shape = Shape(integer()), ambiguous = ambiguous_abstract(update))
  )

  update_computation_graph <- trace_fn(update_computation, dummy_args, desc = desc_update)

  # Register constants from the update computation graph
  for (const in update_computation_graph$constants) {
    get_box_or_register_const(current_desc, const)
  }

  infer_fn <- function(
    input,
    scatter_indices,
    update,
    update_window_dims,
    inserted_window_dims,
    input_batching_dims,
    scatter_indices_batching_dims,
    scatter_dims_to_operand_dims,
    index_vector_dim,
    indices_are_sorted,
    unique_indices,
    update_computation_graph
  ) {
    # Convert 1-based dimension numbers to 0-based
    scatter_dimension_numbers <- stablehlo::ScatterDimensionNumbers(
      update_window_dims = update_window_dims - 1L,
      inserted_window_dims = inserted_window_dims - 1L,
      input_batching_dims = input_batching_dims - 1L,
      scatter_indices_batching_dims = scatter_indices_batching_dims - 1L,
      scatter_dims_to_operand_dims = scatter_dims_to_operand_dims - 1L,
      index_vector_dim = index_vector_dim - 1L
    )

    indices_sorted_attr <- r_to_constant(indices_are_sorted, dtype = "i1", shape = integer())
    unique_indices_attr <- r_to_constant(unique_indices, dtype = "i1", shape = integer())

    out <- stablehlo::infer_types_scatter(
      inputs = list(at2vt(input)),
      scatter_indices = at2vt(scatter_indices),
      updates = list(at2vt(update)),
      scatter_dimension_numbers = scatter_dimension_numbers,
      indices_are_sorted = indices_sorted_attr,
      unique_indices = unique_indices_attr,
      update_computation = stablehlo(update_computation_graph, constants_as_inputs = FALSE)[[1L]]
    )[[1L]]

    out <- vt2at(out)
    out$ambiguous <- input$ambiguous
    list(out)
  }

  out <- graph_desc_add(
    p_scatter,
    args = list(input = input, scatter_indices = scatter_indices, update = update),
    params = list(
      update_window_dims = update_window_dims,
      inserted_window_dims = inserted_window_dims,
      input_batching_dims = input_batching_dims,
      scatter_indices_batching_dims = scatter_indices_batching_dims,
      scatter_dims_to_operand_dims = scatter_dims_to_operand_dims,
      index_vector_dim = index_vector_dim,
      indices_are_sorted = indices_are_sorted,
      unique_indices = unique_indices,
      update_computation_graph = update_computation_graph
    ),
    infer_fn = infer_fn,
    desc = current_desc,
    debug_mode = debug_mode
  )

  out[[1L]]
}

p_gather <- AnvilPrimitive("gather")
#' @title Primitive Gather
#' @description
#' Gathers slices from the operand at positions specified by start_indices.
#' @template param_operand
#' @param start_indices ([`tensorish`] of integer type)\cr
#'   Starting indices for the gather operation.
#' @param slice_sizes (`integer()`)\cr
#'   The sizes of the slices to gather in each dimension.
#' @param offset_dims (`integer()`)\cr
#'   Dimensions of the operand to gather from.
#' @param collapsed_slice_dims (`integer()`)\cr
#'   Dimensions of the slice to gather.
#' @param operand_batching_dims (`integer()`)\cr
#'   Dimensions of the operand to gather from.
#' @param start_indices_batching_dims (`integer()`)\cr
#'   Dimensions of the start_indices to gather from.
#' @param start_index_map (`integer()`)\cr
#'   Mapping from the start_indices to the operand dimensions.
#' @param index_vector_dim (`integer(1)`)\cr
#'   Dimension of the index vector.
#' @param indices_are_sorted (`logical(1)`)\cr
#'   Whether indices are guaranteed to be sorted.
#' @param unique_indices (`logical(1)`)\cr
#'   Whether indices are guaranteed to be unique (no duplicates).
#' @return [`tensorish`]
#' @section Shapes:
#' See [stablehlo::hlo_gather()] for detailed dimension constraints
#' on `start_indices`, `slice_sizes`, and the dimension mapping
#' parameters.
#' @section StableHLO:
#' Calls [stablehlo::hlo_gather()].
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   operand <- nv_tensor(matrix(1:9, nrow = 3))
#'   indices <- nv_tensor(matrix(c(1L, 3L), ncol = 1))
#'   nvl_gather(
#'     operand, indices,
#'     slice_sizes = c(1L, 3L),
#'     offset_dims = 2L,
#'     collapsed_slice_dims = 1L,
#'     operand_batching_dims = integer(0),
#'     start_indices_batching_dims = integer(0),
#'     start_index_map = 1L,
#'     index_vector_dim = 2L
#'   )
#' })
#' @export
nvl_gather <- function(
  operand,
  start_indices,
  slice_sizes,
  offset_dims,
  collapsed_slice_dims,
  operand_batching_dims,
  start_indices_batching_dims,
  start_index_map,
  index_vector_dim,
  indices_are_sorted = FALSE,
  unique_indices = FALSE
) {
  infer_fn <- function(
    operand,
    start_indices,
    slice_sizes,
    offset_dims,
    collapsed_slice_dims,
    operand_batching_dims,
    start_indices_batching_dims,
    start_index_map,
    index_vector_dim,
    indices_are_sorted,
    unique_indices
  ) {
    gather_dimension_numbers <- stablehlo::GatherDimensionNumbers(
      offset_dims = offset_dims - 1L,
      collapsed_slice_dims = collapsed_slice_dims - 1L,
      operand_batching_dims = operand_batching_dims - 1L,
      start_indices_batching_dims = start_indices_batching_dims - 1L,
      start_index_map = start_index_map - 1L,
      index_vector_dim = index_vector_dim - 1L
    )

    slice_sizes_attr <- r_to_constant(slice_sizes, dtype = "i64", shape = length(slice_sizes))
    indices_sorted_attr <- r_to_constant(indices_are_sorted, dtype = "i1", shape = integer())

    out <- stablehlo::infer_types_gather(
      at2vt(operand),
      at2vt(start_indices),
      gather_dimension_numbers = gather_dimension_numbers,
      slice_sizes = slice_sizes_attr,
      indices_are_sorted = indices_sorted_attr
    )[[1L]]

    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  graph_desc_add(
    p_gather,
    args = list(operand = operand, start_indices = start_indices),
    params = list(
      slice_sizes = slice_sizes,
      offset_dims = offset_dims,
      collapsed_slice_dims = collapsed_slice_dims,
      operand_batching_dims = operand_batching_dims,
      start_indices_batching_dims = start_indices_batching_dims,
      start_index_map = start_index_map,
      index_vector_dim = index_vector_dim,
      indices_are_sorted = indices_are_sorted,
      unique_indices = unique_indices
    ),
    infer_fn = infer_fn
  )[[1L]]
}
