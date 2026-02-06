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
#' Creates a tensor of a given shape and data type, filled with a scalar value.
#' The advantage of using this function instead of e.g. doing
#' `nv_tensor(1, shape = c(100, 100))` is that lowering of [nvl_fill()] is
#' efficiently represented in the compiled program, while the latter uses
#' 100 * 100 * 4 bytes of memory.
#' @param value (`numeric(1)`)\cr
#'   Scalar value to fill the tensor with.
#' @param shape (`integer()`)\cr
#'   Shape of the output tensor.
#' @template param_dtype
#' @template param_ambiguous
#' @return [`tensorish`]\cr
#'   Has the given `shape` and `dtype`.
#' @templateVar primitive_id fill
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_tensor()].
#' @seealso [nv_fill()]
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
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id add
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_add()].
#' @seealso [nv_add()], `+`
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
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id mul
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_multiply()].
#' @seealso [nv_mul()], `*`
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
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id sub
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_subtract()].
#' @seealso [nv_sub()], `-`
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
#' @param operand ([`tensorish`])\cr
#'   Tensorish value of data type integer or floating-point.
#' @template return_prim_unary
#' @templateVar primitive_id negate
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_negate()].
#' @seealso [nv_negate()], unary `-`
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
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id divide
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_divide()].
#' @seealso [nv_div()], `/`
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
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id power
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_power()].
#' @seealso [nv_pow()], `^`
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
#' Broadcasts a tensor to a new shape by replicating the data along new or size-1 dimensions.
#' @template param_prim_operand_any
#' @param shape (`integer()`)\cr
#'   Target shape. Each mapped dimension must either match the corresponding
#'   operand dimension or the operand dimension must be 1.
#' @param broadcast_dimensions (`integer()`)\cr
#'   Maps each dimension of `operand` to a dimension of the output.
#'   Must have length equal to the number of dimensions of `operand`.
#' @return [`tensorish`]\cr
#'   Has the same data type as the input and the given `shape`.
#'   It is ambiguous if the input is ambiguous.
#' @importFrom stablehlo r_to_constant
#' @templateVar primitive_id broadcast_in_dim
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_broadcast_in_dim()].
#' @seealso [nv_broadcast_to()]
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
#' General dot product of two tensors, supporting contraction over arbitrary
#' dimensions and batching.
#' @template params_lhs_rhs
#' @param contracting_dims (`list(integer(), integer())`)\cr
#'   A list of two integer vectors specifying which dimensions of `lhs` and
#'   `rhs` to contract over. The contracted dimensions must have matching sizes.
#' @param batching_dims (`list(integer(), integer())`)\cr
#'   A list of two integer vectors specifying which dimensions of `lhs` and
#'   `rhs` are batch dimensions. These must have matching sizes.
#' @return [`tensorish`]\cr
#'   The output shape is the batch dimensions followed by the remaining
#'   (non-contracted, non-batched) dimensions of `lhs`, then `rhs`.
#' @templateVar primitive_id dot_general
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_dot_general()].
#' @seealso [nv_matmul()], `%*%`
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
#' Permutes the dimensions of a tensor.
#' @template param_prim_operand_any
#' @param permutation (`integer()`)\cr
#'   Specifies the new ordering of dimensions. Must be a permutation of
#'   `seq_len(ndims)` where `ndims` is the number of dimensions of `operand`.
#' @return [`tensorish`]\cr
#'   Has the same data type as the input and shape `nv_shape(operand)[permutation]`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id transpose
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_transpose()].
#' @seealso [nv_transpose()], [t()]
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
#' Reshapes a tensor to a new shape without changing the underlying data.
#' Note that row-major order is used, which differs from R's column-major order.
#' @template param_prim_operand_any
#' @param shape (`integer()`)\cr
#'   Target shape. Must have the same number of elements as `operand`.
#' @return [`tensorish`]\cr
#'   Has the same data type as the input and the given `shape`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id reshape
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_reshape()].
#' @seealso [nv_reshape()]
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
#' @param ... ([`tensorish`])\cr
#'   Tensors to concatenate. Must all have the same data type, ndims,
#'   and shape except along `dimension`.
#' @param dimension (`integer(1)`)\cr
#'   Dimension along which to concatenate (1-indexed).
#' @return [`tensorish`]\cr
#'   Has the same data type as the inputs.
#'   The output shape matches the inputs in all dimensions except `dimension`,
#'   which is the sum of the input sizes along that dimension.
#'   It is ambiguous if all inputs are ambiguous.
#' @templateVar primitive_id concatenate
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_concatenate()].
#' @seealso [nv_concatenate()]
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
#' All indices, limits, and strides are fixed R integers.
#'
#' Use [nvl_dynamic_slice()] instead when the start position must be
#' computed at runtime (e.g. depends on tensor values).
#' @template param_prim_operand_any
#' @param start_indices (`integer()`)\cr
#'   Start indices (inclusive), one per dimension. Must satisfy
#'   `1 <= start_indices <= limit_indices` per dimension.
#' @param limit_indices (`integer()`)\cr
#'   End indices (inclusive), one per dimension. Must satisfy
#'   `limit_indices <= nv_shape(operand)` per dimension.
#' @param strides (`integer()`)\cr
#'   Step sizes, one per dimension. Must be `>= 1`. A stride of `1`
#'   selects every element; a stride of `2` selects every other element, etc.
#' @return [`tensorish`]\cr
#'   Has the same data type as the input and shape
#'   `ceiling((limit_indices - start_indices + 1) / strides)`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id static_slice
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_slice()].
#' @seealso [nvl_dynamic_slice()], [nvl_scatter()], [nvl_gather()], [nv_subset()], `[`
#' @examplesIf pjrt::plugin_is_downloaded()
#' # 1-D: extract elements 2 through 4 (limit is exclusive)
#' jit_eval({
#'   x <- nv_tensor(1:10)
#'   nvl_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 1L)
#' })
#'
#' # 1-D: every other element using strides
#' jit_eval({
#'   x <- nv_tensor(1:10)
#'   nvl_static_slice(x, start_indices = 1L, limit_indices = 10L, strides = 2L)
#' })
#'
#' # 2-D: extract a submatrix (rows 1-2, columns 2-3)
#' jit_eval({
#'   x <- nv_tensor(matrix(1:12, nrow = 3, ncol = 4))
#'   nvl_static_slice(x,
#'     start_indices = c(1L, 2L),
#'     limit_indices = c(3L, 4L),
#'     strides       = c(1L, 1L)
#'   )
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
#' Extracts a slice from a tensor whose start position is determined at
#' runtime via tensor-valued indices. The slice shape (`slice_sizes`) is
#' a fixed R integer vector.
#'
#' Use [nvl_static_slice()] instead when all indices are known at compile
#' time and you need stride support.
#' @template param_prim_operand_any
#' @param ... ([`tensorish`] of integer type)\cr
#'   Scalar start indices, one per dimension. Each must be a
#'   scalar tensor. Pass one scalar per dimension of `operand`.
#' @param slice_sizes (`integer()`)\cr
#'   Size of the slice in each dimension. Must have length equal to
#'   `ndims(operand)` and satisfy `1 <= slice_sizes <= nv_shape(operand)`
#'   per dimension.
#' @section Out Of Bounds Behavior:
#' Start indices are clamped before the slice is extracted:
#' `adjusted_start_indices = clamp(1, start_indices, nv_shape(operand) - slice_sizes + 1)`.
#' This means that out-of-bounds indices will not cause an error, but
#' the effective start position may differ from the requested one.
#' @return [`tensorish`]\cr
#'   Has the same data type as the input and shape `slice_sizes`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id dynamic_slice
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_dynamic_slice()].
#' @seealso [nvl_static_slice()], [nvl_dynamic_update_slice()], [nvl_scatter()], [nvl_gather()], [nv_subset()], `[`
#' @examplesIf pjrt::plugin_is_downloaded()
#' # 1-D: extract 3 elements starting at position 3
#' jit_eval({
#'   x <- nv_tensor(1:10)
#'   start <- nv_scalar(3L)
#'   nvl_dynamic_slice(x, start, slice_sizes = 3L)
#' })
#'
#' # 2-D: extract a 2x2 block from a matrix
#' jit_eval({
#'   x <- nv_tensor(matrix(1:12, nrow = 3, ncol = 4))
#'   row_start <- nv_scalar(2L)
#'   col_start <- nv_scalar(1L)
#'   nvl_dynamic_slice(x, row_start, col_start, slice_sizes = c(2L, 2L))
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
#' Returns a copy of `operand` with a slice replaced by `update` at a
#' runtime-determined position. This is the write counterpart of
#' [nvl_dynamic_slice()]: dynamic slice reads a block from a tensor,
#' while dynamic update slice writes a block into a tensor.
#' @template param_prim_operand_any
#' @param update ([`tensorish`])\cr
#'   The values to write at the specified position. Must have the same
#'   data type and number of dimensions as `operand`, with
#'   `nv_shape(update) <= nv_shape(operand)` per dimension.
#' @param ... ([`tensorish`] of integer type)\cr
#'   Scalar start indices, one per dimension of `operand`.
#'   Each must be a scalar tensor.
#' @inheritSection nvl_dynamic_slice Out Of Bounds Behavior
#' @return [`tensorish`]\cr
#'   Has the same data type and shape as `operand`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id dynamic_update_slice
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_dynamic_update_slice()].
#' @seealso [nvl_dynamic_slice()], [nvl_scatter()], [nvl_gather()], [nv_subset_assign()], `[<-`
#' @examplesIf pjrt::plugin_is_downloaded()
#' # 1-D: overwrite two elements starting at position 2
#' jit_eval({
#'   x <- nv_tensor(1:5)
#'   update <- nv_tensor(c(10L, 20L))
#'   start <- nv_scalar(2L)
#'   nvl_dynamic_update_slice(x, update, start)
#' })
#'
#' # 2-D: write a 2x2 block into a 3x4 matrix
#' jit_eval({
#'   x <- nv_tensor(matrix(0L, nrow = 3, ncol = 4))
#'   update <- nv_tensor(matrix(c(1L, 2L, 3L, 4L), nrow = 2, ncol = 2))
#'   row_start <- nv_scalar(2L)
#'   col_start <- nv_scalar(3L)
#'   nvl_dynamic_update_slice(x, update, row_start, col_start)
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
#' @seealso [nv_reduce_sum()]
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
#' @seealso [nv_reduce_prod()]
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
#' @seealso [nv_reduce_max()]
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
#' @seealso [nv_reduce_min()]
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
#' @seealso [nv_reduce_any()]
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
#' @seealso [nv_reduce_all()]
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
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "EQ"`.
#' @seealso [nv_eq()], `==`
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
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id not_equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "NE"`.
#' @seealso [nv_ne()], `!=`
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
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id greater
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "GT"`.
#' @seealso [nv_gt()], `>`
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
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id greater_equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "GE"`.
#' @seealso [nv_ge()], `>=`
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
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id less
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "LT"`.
#' @seealso [nv_lt()], `<`
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
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id less_equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_compare()] with `comparison_direction = "LE"`.
#' @seealso [nv_le()], `<=`
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
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id maximum
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_maximum()].
#' @seealso [nv_max()]
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
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id minimum
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_minimum()].
#' @seealso [nv_min()]
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
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id remainder
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_remainder()].
#' @seealso [nv_remainder()], `%%`
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
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id and
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_and()].
#' @seealso [nv_and()], `&`
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
#' @param operand ([`tensorish`])\cr
#'   Tensorish value of data type boolean, integer, or unsigned integer.
#' @template return_prim_unary
#' @templateVar primitive_id not
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_not()].
#' @seealso [nv_not()]
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
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id or
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_or()].
#' @seealso [nv_or()], `|`
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
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id xor
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_xor()].
#' @seealso [nv_xor()]
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
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id shift_left
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_shift_left()].
#' @seealso [nv_shift_left()]
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
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id shift_right_logical
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_shift_right_logical()].
#' @seealso [nv_shift_right_logical()]
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
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id shift_right_arithmetic
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_shift_right_arithmetic()].
#' @seealso [nv_shift_right_arithmetic()]
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
#' @template params_prim_lhs_rhs_float
#' @template return_prim_binary
#' @templateVar primitive_id atan2
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_atan2()].
#' @seealso [nv_atan2()]
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
#' Reinterprets the bits of a tensor as a different data type without
#' modifying the underlying data.
#' @template param_prim_operand_any
#' @param dtype (`character(1)` | [`TensorDataType`])\cr
#'   Target data type. If it has the same bit width as the input, the output
#'   shape is unchanged. If narrower, an extra trailing dimension is added.
#'   If wider, the last dimension is consumed.
#' @return [`tensorish`]\cr
#'   Has the given `dtype`.
#' @templateVar primitive_id bitcast_convert
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_bitcast_convert()].
#' @seealso [nv_bitcast_convert()]
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(1L)
#'   nvl_bitcast_convert(x, dtype = "i8")
#' })
#' jit_eval({
#'   x <- nv_tensor(rep(1L, 4), dtype = "i8")
#'   nvl_bitcast_convert(x, dtype = "i32")
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
#' @template param_prim_operand_signed_numeric
#' @template return_prim_unary
#' @templateVar primitive_id abs
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_abs()].
#' @seealso [nv_abs()], [abs()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id sqrt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_sqrt()].
#' @seealso [nv_sqrt()], [sqrt()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id rsqrt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_rsqrt()].
#' @seealso [nv_rsqrt()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id log
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_log()].
#' @seealso [nv_log()], [log()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id tanh
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_tanh()].
#' @seealso [nv_tanh()], [tanh()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id tan
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_tan()].
#' @seealso [nv_tan()], [tan()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id sine
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_sine()].
#' @seealso [nv_sine()], [sin()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id cosine
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_cosine()].
#' @seealso [nv_cosine()], [cos()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id floor
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_floor()].
#' @seealso [nv_floor()], [floor()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id ceil
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_ceil()].
#' @seealso [nv_ceil()], [ceiling()]
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
#' @template param_prim_operand_signed_numeric
#' @template return_prim_unary
#' @templateVar primitive_id sign
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_sign()].
#' @seealso [nv_sign()], [sign()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id exp
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_exponential()].
#' @seealso [nv_exp()], [exp()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id expm1
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_exponential_minus_one()].
#' @seealso [nv_expm1()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id log1p
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_log_plus_one()].
#' @seealso [nv_log1p()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id cbrt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_cbrt()].
#' @seealso [nv_cbrt()]
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
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id logistic
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_logistic()].
#' @seealso [nv_logistic()]
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
#' @template param_prim_operand_float
#' @return [`tensorish`]\cr
#'   Has the same shape as the input and boolean data type.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id is_finite
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_is_finite()].
#' @seealso [nv_is_finite()]
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
#' @param operand ([`tensorish`])\cr
#'   Tensorish value of data type integer or unsigned integer.
#' @template return_prim_unary
#' @templateVar primitive_id popcnt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_popcnt()].
#' @seealso [nv_popcnt()]
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
#' Clamps every element of `operand` to the range `[min_val, max_val]`,
#' i.e. `max(min_val, min(operand, max_val))`.
#' @param min_val ([`tensorish`])\cr
#'   Minimum value. Must be scalar or the same shape as `operand`.
#' @template param_prim_operand_any
#' @param max_val ([`tensorish`])\cr
#'   Maximum value. Must be scalar or the same shape as `operand`.
#' @return [`tensorish`]\cr
#'   Has the same data type and shape as `operand`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id clamp
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_clamp()].
#' @seealso [nv_clamp()]
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
#' @template param_prim_operand_any
#' @param dims (`integer()`)\cr
#'   Dimensions to reverse (1-indexed).
#' @return [`tensorish`]\cr
#'   Has the same data type and shape as `operand`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id reverse
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_reverse()].
#' @seealso [nv_reverse()]
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
#' @return [`tensorish`]\cr
#'   Has the given `dtype` and `shape`.
#' @templateVar primitive_id iota
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_iota()].
#' @seealso [nv_iota()]
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
#' @template param_prim_operand_any
#' @param padding_value ([`tensorish`])\cr
#'   Scalar value to use for padding. Must have the same dtype as `operand`.
#' @param edge_padding_low (`integer()`)\cr
#'   Amount of padding to add at the start of each dimension.
#' @param edge_padding_high (`integer()`)\cr
#'   Amount of padding to add at the end of each dimension.
#' @param interior_padding (`integer()`)\cr
#'   Amount of padding to add between elements in each dimension.
#' @return [`tensorish`]\cr
#'   Has the same data type as `operand`.
#'   For the output shape see the underlying stablehlo documentation ([stablehlo::hlo_pad()]).
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id pad
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_pad()].
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
#' Rounds the elements of a tensor to the nearest integer.
#' @template param_prim_operand_float
#' @param method (`character(1)`)\cr
#'   Rounding method. `"nearest_even"` (default) rounds to the nearest even
#'   integer on a tie, `"afz"` rounds away from zero on a tie.
#' @return [`tensorish`]\cr
#'   Has the same dtype and shape as `operand`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id round
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_round_nearest_even()] or
#' [stablehlo::hlo_round_nearest_afz()] depending on the `method` parameter.
#' @seealso [nv_round()]
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
#' Converts the elements of a tensor to a different data type.
#' @template param_prim_operand_any
#' @param dtype (`character(1)` | [`TensorDataType`])\cr
#'   Target data type.
#' @template param_ambiguous
#' @return [`tensorish`]\cr
#'   Has the given `dtype` and the same shape as `operand`.
#'   Ambiguity is controlled by the `ambiguous` parameter.
#' @templateVar primitive_id convert
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_convert()].
#' @seealso [nv_convert()]
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
#' Element-wise selection based on a boolean predicate, like R's [ifelse()].
#' For each element, returns the corresponding element from `true_value` where
#' `pred` is `TRUE` and from `false_value` where `pred` is `FALSE`.
#' @param pred ([`tensorish`] of boolean type)\cr
#'   Predicate tensor. Must be scalar or have the same shape as
#'   `true_value`.
#' @param true_value,false_value ([`tensorish`])\cr
#'   Values to select from. Must have the same dtype and shape.
#' @return [`tensorish`]\cr
#'   Has the same dtype and shape as `true_value`.
#'   It is ambiguous if both `true_value` and `false_value` are ambiguous.
#' @templateVar primitive_id select
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_select()].
#' @seealso [nv_ifelse()]
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
#' Conditional execution of one of two branches based on a scalar boolean
#' predicate. Unlike [nvl_ifelse()] which operates element-wise, this
#' evaluates only the selected branch.
#' @param pred ([`tensorish`])\cr
#'   Scalar boolean predicate that determines which branch to execute.
#' @param true,false (NSE)\cr
#'   Expressions for the true and false branches. Both must return outputs
#'   with the same structure, dtypes, and shapes.
#' @return Result of the executed branch.\cr
#'   An output is ambiguous if it is ambiguous in both branches.
#' @templateVar primitive_id if
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_if()].
#' @seealso [nv_if()], [nvl_ifelse()]
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
#' Repeatedly executes `body` while `cond` returns `TRUE`, like R's
#' `while` loop. The loop state is initialized with `init` and
#' passed through each iteration.
#' Otherwise, no state is maintained between iterations.
#' @param init (`named list()`)\cr
#'   Named list of initial state values.
#' @param cond (`function`)\cr
#'   Condition function that receives the current state as arguments
#'   and outputs whether to continue the loop.
#' @param body (`function`)\cr
#'   Body function that receives the current state as arguments and
#'   returns a named list with the same structure, dtypes, and shapes
#'   as `init`.
#' @return Named list with the same structure as `init` containing the
#'   final state after the loop terminates.
#' @templateVar primitive_id while
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_while()].
#' @seealso [nv_while()]
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   nvl_while(
#'     init = list(i = 0L, total = 0L),
#'     cond = function(i, total) i <= 5L,
#'     body = function(i, total) list(
#'       i = i + 1L,
#'       total = total + i
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
#' Prints a tensor value to the console during execution and returns the
#' input unchanged. This is useful for debugging JIT-compiled code.
#' @note Currently only works on the CPU backend.
#' @template param_prim_operand_any
#' @return [`tensorish`]\cr
#'   Returns `operand` as-is.
#' @templateVar primitive_id print
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_custom_call()].
#' @seealso [nv_print()]
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_tensor(c(1, 2, 3), device = "cpu")
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
#' Generates pseudo-random numbers using the specified algorithm and returns
#' the updated RNG state together with the generated values.
#' @template param_initial_state
#' @param rng_algorithm (`character(1)`)\cr
#'   RNG algorithm name. Default is `"THREE_FRY"`.
#' @param dtype (`character(1)` | [`TensorDataType`])\cr
#'   Data type of the generated random values.
#' @template param_shape
#' @return `list` of two [`tensorish`] values:\cr
#'   The first element is the updated RNG state with the same dtype and shape
#'   as `initial_state`. The second element is a tensor of random values with
#'   the given `dtype` and `shape`.
#' @templateVar primitive_id rng_bit_generator
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_rng_bit_generator()].
#' @seealso [nv_runif()], [nv_rnorm()]
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   state <- nv_tensor(c(0L, 0L), dtype = "ui64")
#'   nvl_rng_bit_generator(state, dtype = "f32", shape = c(3, 2))
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
#' Produces a result tensor identical to `input` except that slices at
#' positions specified by `scatter_indices` are updated with values from
#' the `update` tensor. When multiple indices point to the same location,
#' the `update_computation` function determines how to combine the values
#' (by default the new value replaces the old one).
#'
#' This is the inverse of [nvl_gather()]: gather reads slices from a tensor
#' at given indices, while scatter writes slices into a tensor at given
#' indices.
#' @param input ([`tensorish`])\cr
#'   Tensorish value of any data type. The base tensor to scatter into.
#' @param scatter_indices ([`tensorish`] of integer type)\cr
#'   Tensor of indices. Contains index vectors that map to positions in
#'   `input` via `scatter_dims_to_operand_dims`. The dimension specified
#'   by `index_vector_dim` holds the index vectors.
#' @param update ([`tensorish`])\cr
#'   Update values tensor. Must have the same data type as `input`.
#' @param update_window_dims (`integer()`)\cr
#'   Dimensions of `update` that are window dimensions, i.e. they
#'   correspond to the slice being written into `input`.
#' @param inserted_window_dims (`integer()`)\cr
#'   Dimensions of `input` whose slices have size 1 and are inserted
#'   (not present) in the `update` window. Together with
#'   `update_window_dims` and `input_batching_dims`, these must account
#'   for all dimensions of `input`.
#' @param input_batching_dims (`integer()`)\cr
#'   Dimensions of `input` that are batch dimensions.
#'   Use `integer(0)` when there are no batch dimensions.
#' @param scatter_indices_batching_dims (`integer()`)\cr
#'   Dimensions of `scatter_indices` that correspond to batch
#'   dimensions. Must have the same length as `input_batching_dims`.
#' @param scatter_dims_to_operand_dims (`integer()`)\cr
#'   Maps each component of the index vector to an `input` dimension.
#'   For example, `scatter_dims_to_operand_dims = c(1L)` means each
#'   index vector indexes into the first dimension of `input`.
#' @param index_vector_dim (`integer(1)`)\cr
#'   Dimension of `scatter_indices` that contains the index vectors.
#'   If set to `ndims(scatter_indices) + 1`, each scalar element of
#'   `scatter_indices` is treated as a length-1 index vector.
#' @param indices_are_sorted (`logical(1)`)\cr
#'   Whether indices are guaranteed to be sorted. Setting to `TRUE`
#'   may improve performance but produces undefined behavior if the
#'   indices are not actually sorted. Default `FALSE`.
#' @param unique_indices (`logical(1)`)\cr
#'   Whether indices are guaranteed to be unique (no duplicates).
#'   Setting to `TRUE` may improve performance but produces undefined
#'   behavior if the indices are not actually unique. Default `FALSE`.
#' @param update_computation (`function`)\cr
#'   Binary function `f(old, new)` that combines the existing value in
#'   `input` with the value from `update`. The default (`NULL`) uses
#'   `function(old, new) new`, which replaces the old value.
#' @return [`tensorish`]\cr
#'   Has the same data type and shape as `input`.
#'   It is ambiguous if `input` is ambiguous.
#' @section Out Of Bounds Behavior:
#' If a computed result index falls outside the bounds of `input`, the
#' update for that index is silently ignored.
#' @section Update Order:
#' When multiple indices in `scatter_indices` map to the same element
#' of `input`, the order in which `update_computation` is applied is
#' implementation-defined and may vary between plugins ("cpu", "cuda").
#' @templateVar primitive_id scatter
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_scatter()].
#' @seealso [nvl_gather()], [nv_subset()], [nv_subset_assign()], `[`, `[<-`
#' @examplesIf pjrt::plugin_is_downloaded()
#' # Scatter values 10 and 30 into positions 1 and 3 of a zero vector
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
#' Gathers slices from the `operand` tensor at positions specified by
#' `start_indices`. Each index vector in `start_indices` identifies a
#' starting position in `operand`, and a slice of size `slice_sizes` is
#' extracted from that position. The gathered slices are assembled into
#' the output tensor.
#'
#' This is the inverse of [nvl_scatter()]: gather reads slices from a
#' tensor at given indices, while scatter writes slices into a tensor at
#' given indices.
#' @template param_prim_operand_any
#' @param start_indices ([`tensorish`] of integer type)\cr
#'   Tensor of starting indices. Contains index vectors that map to
#'   positions in `operand` via `start_index_map`. The dimension
#'   specified by `index_vector_dim` holds the index vectors.
#' @param slice_sizes (`integer()`)\cr
#'   Size of the slice to gather from `operand` in each dimension.
#'   Must have length equal to `ndims(operand)`.
#' @param offset_dims (`integer()`)\cr
#'   Dimensions in the output that correspond to the non-collapsed
#'   slice dimensions of `operand`.
#' @param collapsed_slice_dims (`integer()`)\cr
#'   Dimensions of `operand` that are collapsed (removed) from the
#'   slice. The corresponding entries in `slice_sizes` must be `1`.
#'   Together with `offset_dims` and `operand_batching_dims`, these
#'   must account for all dimensions of `operand`.
#' @param operand_batching_dims (`integer()`)\cr
#'   Dimensions of `operand` that are batch dimensions.
#'   Use `integer(0)` when there are no batch dimensions.
#' @param start_indices_batching_dims (`integer()`)\cr
#'   Dimensions of `start_indices` that correspond to batch
#'   dimensions. Must have the same length as `operand_batching_dims`.
#' @param start_index_map (`integer()`)\cr
#'   Maps each component of the index vector to an `operand`
#'   dimension. For example, `start_index_map = c(1L)` means each
#'   index vector indexes into the first dimension of `operand`.
#' @param index_vector_dim (`integer(1)`)\cr
#'   Dimension of `start_indices` that contains the index vectors.
#'   If set to `ndims(start_indices) + 1`, each scalar element of
#'   `start_indices` is treated as a length-1 index vector.
#' @param indices_are_sorted (`logical(1)`)\cr
#'   Whether indices are guaranteed to be sorted. Setting to `TRUE`
#'   may improve performance but produces undefined behavior if the
#'   indices are not actually sorted. Default `FALSE`.
#' @param unique_indices (`logical(1)`)\cr
#'   Whether indices are guaranteed to be unique (no duplicates).
#'   Setting to `TRUE` may improve performance but produces undefined
#'   behavior if the indices are not actually unique. Default `FALSE`.
#' @return [`tensorish`]\cr
#'   Has the same data type as `operand`. The output shape is composed
#'   of the offset dimensions (from the slice) and the remaining
#'   dimensions from `start_indices`. See the underluing stableHLO function
#'   for more details.
#' @section Out Of Bounds Behavior:
#' Start indices are clamped before the slice is extracted:
#' `clamp(1, start_index, nv_shape(operand) - slice_sizes + 1)`.
#' This means that out-of-bounds indices will not cause an error, but
#' the effective start position may differ from the requested one.
#' @templateVar primitive_id gather
#' @template section_rules
#' @section StableHLO:
#' Lowers to [stablehlo::hlo_gather()].
#' @seealso [nvl_scatter()], [nv_subset()], [nv_subset_assign()], `[`, `[<-`
#' @examplesIf pjrt::plugin_is_downloaded()
#' # Gather rows 1 and 3 from a 3x3 matrix
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
