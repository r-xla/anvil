#' @include primitives.R

# TODO: Here we don't have to re-do the type inference again, because it was already done.

p_fill[["stablehlo"]] <- function(value, shape, dtype, ambiguous) {
  # ambiguity only relevant for type promotion, but when we lower
  # there is no type promotion, so it has no effect
  list(stablehlo::hlo_tensor(value, shape = shape, dtype = dtype))
}

p_add[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_add(lhs, rhs))
}

p_mul[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_multiply(lhs, rhs))
}

p_sub[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_subtract(lhs, rhs))
}

p_negate[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_negate(operand))
}

p_div[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_divide(lhs, rhs))
}

p_pow[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_power(lhs, rhs))
}

p_broadcast_in_dim[["stablehlo"]] <- function(operand, shape, broadcast_dimensions) {
  list(stablehlo::hlo_broadcast_in_dim(operand, broadcast_dimensions - 1L, shape))
}

p_dot_general[["stablehlo"]] <- function(lhs, rhs, contracting_dims, batching_dims) {
  contracting_dims <- lapply(contracting_dims, \(x) x - 1L)
  batching_dims <- lapply(batching_dims, \(x) x - 1L)
  list(stablehlo::hlo_dot_general(lhs, rhs, contracting_dims, batching_dims))
}

p_transpose[["stablehlo"]] <- function(operand, permutation) {
  list(stablehlo::hlo_transpose(operand, permutation - 1L))
}

p_reshape[["stablehlo"]] <- function(operand, shape) {
  list(stablehlo::hlo_reshape(operand, shape))
}

p_concatenate[["stablehlo"]] <- function(..., dimension) {
  list(stablehlo::hlo_concatenate(..., dimension = dimension - 1L))
}

p_static_slice[["stablehlo"]] <- function(operand, start_indices, limit_indices, strides) {
  # we use 1:n, which includes n, but this translates to 0:n in stablehlo
  list(stablehlo::hlo_slice(operand, start_indices - 1L, limit_indices, strides))
}

p_dynamic_slice[["stablehlo"]] <- function(operand, ..., slice_sizes) {
  start_indices <- list(...)
  # Convert start indices from 1-based to 0-based by subtracting 1
  start_indices_0based <- lapply(start_indices, function(idx) {
    one <- stablehlo::hlo_scalar(1L, dtype = dtype(idx), func = idx$func)
    stablehlo::hlo_subtract(idx, one)
  })
  list(rlang::exec(
    stablehlo::hlo_dynamic_slice,
    operand,
    !!!start_indices_0based,
    slice_sizes = slice_sizes
  ))
}

p_dynamic_update_slice[["stablehlo"]] <- function(operand, update, ...) {
  start_indices <- list(...)
  if (!length(start_indices)) {
    return(list(update))
  }
  # Convert start indices from 1-based to 0-based by subtracting 1
  start_indices_0based <- lapply(start_indices, function(idx) {
    one <- stablehlo::hlo_scalar(1L, dtype = dtype(idx), func = idx$func)
    stablehlo::hlo_subtract(idx, one)
  })
  list(rlang::exec(
    stablehlo::hlo_dynamic_update_slice,
    operand,
    update,
    !!!start_indices_0based
  ))
}


.stablehlo_apply_reduce <- function(reductor, operand, init, dims, drop) {
  local_func("")
  dt <- as.character(operand$value_type$type$dtype)
  f <- hlo_return(reductor(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))
  out <- stablehlo::hlo_reduce(list(operand), list(init(operand)), dims - 1L, f)

  if (drop) {
    return(list(out))
  }

  shape_out <- shape(operand$value_type)
  shape_out[dims] <- 1L
  list(stablehlo::hlo_reshape(out, shape_out))
}

p_reduce_sum[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(0, dtype = dtype(operand), func = operand$func)
  }
  .stablehlo_apply_reduce(stablehlo::hlo_add, operand, init, dims, drop)
}

p_reduce_prod[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(1, dtype = dtype(operand), func = operand$func)
  }
  .stablehlo_apply_reduce(stablehlo::hlo_multiply, operand, init, dims, drop)
}


p_reduce_max[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_minval(dtype(operand), "cpu"))
  }
  .stablehlo_apply_reduce(stablehlo::hlo_maximum, operand, init, dims, drop)
}

p_reduce_min[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_maxval(dtype(operand), "cpu"))
  }
  .stablehlo_apply_reduce(stablehlo::hlo_minimum, operand, init, dims, drop)
}

p_reduce_any[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(FALSE)
  }
  .stablehlo_apply_reduce(stablehlo::hlo_or, operand, init, dims, drop)
}

p_reduce_all[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(TRUE)
  }
  .stablehlo_apply_reduce(stablehlo::hlo_and, operand, init, dims, drop)
}

.stablehlo_argminmax <- function(direction) {
  function(operand, dim, drop, index_dtype) {
    val_dt <- as.character(dtype(operand))
    idx_dt <- as.character(index_dtype)
    op_shape <- shape(operand$value_type)

    # Build comparator: (val_l, val_r, idx_l, idx_r) -> (sel_val, sel_idx)
    local_func("")
    val_l <- hlo_input("val_l", val_dt)
    val_r <- hlo_input("val_r", val_dt)
    idx_l <- hlo_input("idx_l", idx_dt)
    idx_r <- hlo_input("idx_r", idx_dt)

    ct <- if (inherits(operand$value_type$type$dtype, "FloatType")) {
      "FLOAT"
    } else if (inherits(operand$value_type$type$dtype, "IntegerType")) {
      "SIGNED"
    } else {
      "UNSIGNED"
    }
    cmp <- stablehlo::hlo_compare(val_l, val_r, comparison_direction = direction, compare_type = ct)
    sel_val <- stablehlo::hlo_select(cmp, val_l, val_r)
    sel_idx <- stablehlo::hlo_select(cmp, idx_l, idx_r)
    body <- hlo_return(sel_val, sel_idx)

    # Init values
    init_val <- if (direction == "LE") {
      hlo_scalar(nv_maxval(val_dt, "cpu"))
    } else {
      hlo_scalar(nv_minval(val_dt, "cpu"))
    }
    init_idx <- hlo_scalar(0L, dtype = idx_dt, func = operand$func)

    # Iota for 0-based indices along dim
    indices <- stablehlo::hlo_iota(iota_dimension = dim - 1L, dtype = idx_dt, shape = op_shape)

    results <- stablehlo::hlo_reduce(
      list(operand, indices),
      list(init_val, init_idx),
      dim - 1L,
      body
    )

    # Add 1 for 1-based indexing
    idx_out <- results[[2L]]
    one <- stablehlo::hlo_scalar(1L, dtype = idx_dt, func = operand$func)
    one_bc <- stablehlo::hlo_broadcast_in_dim(one, integer(0), shape(idx_out$value_type))
    idx_out <- stablehlo::hlo_add(idx_out, one_bc)

    if (!drop) {
      out_shape <- op_shape
      out_shape[dim] <- 1L
      idx_out <- stablehlo::hlo_reshape(idx_out, out_shape)
    }
    list(idx_out)
  }
}

p_argmin[["stablehlo"]] <- .stablehlo_argminmax("LE")
p_argmax[["stablehlo"]] <- .stablehlo_argminmax("GE")

p_reduce[["stablehlo"]] <- function(operand, init_value, body, dims, drop) {
  dt <- as.character(dtype(operand))

  local_func("")
  acc <- hlo_input("acc", dt)
  new <- hlo_input("new", dt)
  body_func <- hlo_return(body(acc, new))

  out <- stablehlo::hlo_reduce(list(operand), list(init_value), dims - 1L, body_func)

  if (drop) {
    return(list(out))
  }

  shape_out <- shape(operand$value_type)
  shape_out[dims] <- 1L
  list(stablehlo::hlo_reshape(out, shape_out))
}

# comparison jit rules ----------------------------------------------------------

.compare_type_for <- function(vt) {
  dt <- vt$value_type$type$dtype
  if (inherits(dt, "FloatType")) {
    "FLOAT"
  } else if (inherits(dt, "IntegerType")) {
    "SIGNED"
  } else if (inherits(dt, "UIntegerType") || inherits(dt, "BooleanType")) {
    "UNSIGNED"
  } else {
    cli_abort("Unsupported dtype for compare")
  }
}

.stablehlo_compare_bin <- function(direction) {
  function(lhs, rhs) {
    ct <- .compare_type_for(lhs)
    list(stablehlo::hlo_compare(lhs, rhs, comparison_direction = direction, compare_type = ct))
  }
}

p_eq[["stablehlo"]] <- .stablehlo_compare_bin("EQ")
p_ne[["stablehlo"]] <- .stablehlo_compare_bin("NE")
p_gt[["stablehlo"]] <- .stablehlo_compare_bin("GT")
p_ge[["stablehlo"]] <- .stablehlo_compare_bin("GE")
p_lt[["stablehlo"]] <- .stablehlo_compare_bin("LT")
p_le[["stablehlo"]] <- .stablehlo_compare_bin("LE")


# binary simple math jit rules ---------------------------------------------------

p_max[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_maximum(lhs, rhs))
}

p_min[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_minimum(lhs, rhs))
}

p_remainder[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_remainder(lhs, rhs))
}

p_and[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_and(lhs, rhs))
}

p_not[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_not(operand))
}

p_or[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_or(lhs, rhs))
}

p_xor[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_xor(lhs, rhs))
}

p_shift_left[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_shift_left(lhs, rhs))
}

p_shift_right_logical[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_shift_right_logical(lhs, rhs))
}

p_shift_right_arithmetic[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_shift_right_arithmetic(lhs, rhs))
}

p_atan2[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_atan2(lhs, rhs))
}

p_bitcast_convert[["stablehlo"]] <- function(operand, dtype) {
  list(stablehlo::hlo_bitcast_convert(operand, dtype))
}

# unary simple math jit rules ---------------------------------------------------

p_abs[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_abs(operand))
}

p_sqrt[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_sqrt(operand))
}

p_rsqrt[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_rsqrt(operand))
}

p_log[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_log(operand))
}

p_tanh[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_tanh(operand))
}

p_tan[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_tan(operand))
}

p_sine[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_sine(operand))
}

p_cosine[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_cosine(operand))
}

p_floor[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_floor(operand))
}

p_ceil[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_ceil(operand))
}

p_sign[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_sign(operand))
}

p_exp[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_exponential(operand))
}

p_expm1[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_exponential_minus_one(operand))
}

p_log1p[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_log_plus_one(operand))
}

p_cbrt[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_cbrt(operand))
}

p_logistic[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_logistic(operand))
}

p_is_finite[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_is_finite(operand))
}

p_popcnt[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_popcnt(operand))
}

p_clamp[["stablehlo"]] <- function(min_val, operand, max_val) {
  list(stablehlo::hlo_clamp(min_val, operand, max_val))
}

p_reverse[["stablehlo"]] <- function(operand, dims) {
  list(stablehlo::hlo_reverse(operand, dims - 1L))
}

p_iota[["stablehlo"]] <- function(dim, dtype, shape, start, ambiguous) {
  out <- stablehlo::hlo_iota(iota_dimension = dim - 1L, dtype = dtype, shape = shape)
  if (start != 0L) {
    offset <- stablehlo::hlo_broadcast_in_dim(
      stablehlo::hlo_scalar(start, dtype = dtype, func = out$func),
      integer(0),
      shape
    )
    out <- stablehlo::hlo_add(out, offset)
  }
  list(out)
}

p_pad[["stablehlo"]] <- function(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding) {
  list(stablehlo::hlo_pad(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding))
}

p_round[["stablehlo"]] <- function(operand, method) {
  switch(
    method,
    afz = list(stablehlo::hlo_round_nearest_afz(operand)),
    nearest_even = list(stablehlo::hlo_round_nearest_even(operand)),
    cli_abort("invalid method: {method}")
  )
}

p_convert[["stablehlo"]] <- function(operand, dtype, ambiguous) {
  list(stablehlo::hlo_convert(operand, dtype))
}


p_select[["stablehlo"]] <- function(pred, true_value, false_value) {
  list(stablehlo::hlo_select(pred, true_value, false_value))
}

# RNG jit rules --------------------------------------------------------

p_rng_bit_generator[["stablehlo"]] <- function(initial_state, rng_algorithm, dtype, shape) {
  stablehlo::hlo_rng_bit_generator(initial_state, rng_algorithm, dtype, shape)
}

p_print[["stablehlo"]] <- function(operand, footer) {
  backend_config <- stablehlo::CustomOpBackendConfig(list(
    stablehlo::StringAttr(name = "print_header", value = "AnvilArray"),
    stablehlo::StringAttr(name = "print_footer", value = footer)
  ))

  # has side-effect
  stablehlo::hlo_custom_call(
    operand,
    call_target_name = "print_tensor",
    api_version = 4L,
    has_side_effect = TRUE,
    backend_config = backend_config
  )
  # we just return the input
  list(operand)
}

# higher order primitives --------------------------------------------------------

p_if[["stablehlo"]] <- function(pred, true_graph, false_graph, .env) {
  true_func <- stablehlo(true_graph, constants_as_inputs = FALSE, env = .env)[[1L]]
  false_func <- stablehlo(false_graph, constants_as_inputs = FALSE, env = .env)[[1L]]
  stablehlo::hlo_if(pred, true_func, false_func, simplify = FALSE)
}

p_while[["stablehlo"]] <- function(..., cond_graph, body_graph, .env) {
  body_func <- stablehlo(body_graph, constants_as_inputs = FALSE, env = .env)[[1L]]
  cond_func <- stablehlo(cond_graph, constants_as_inputs = FALSE, env = .env)[[1L]]
  stablehlo::hlo_while(..., cond = cond_func, body = body_func, simplify = FALSE)
}

p_scatter[["stablehlo"]] <- function(
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
  update_computation_graph,
  .env
) {
  update_func <- stablehlo(update_computation_graph, constants_as_inputs = FALSE, env = .env)[[1L]]

  scatter_dimension_numbers <- stablehlo::ScatterDimensionNumbers(
    update_window_dims = update_window_dims - 1L,
    inserted_window_dims = inserted_window_dims - 1L,
    input_batching_dims = input_batching_dims - 1L,
    scatter_indices_batching_dims = scatter_indices_batching_dims - 1L,
    scatter_dims_to_operand_dims = scatter_dims_to_operand_dims - 1L,
    index_vector_dim = index_vector_dim - 1L
  )

  one <- stablehlo::hlo_tensor(1L, shape = shape(scatter_indices), dtype = dtype(scatter_indices))
  scatter_indices_0based <- stablehlo::hlo_subtract(scatter_indices, one)

  result <- stablehlo::hlo_scatter(
    inputs = list(input),
    scatter_indices = scatter_indices_0based,
    updates = list(update),
    scatter_dimension_numbers = scatter_dimension_numbers,
    indices_are_sorted = indices_are_sorted,
    unique_indices = unique_indices,
    update_computation = update_func
  )

  list(result)
}

p_gather[["stablehlo"]] <- function(
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
  # Convert 1-based dimension numbers to 0-based for stablehlo
  gdn_0based <- stablehlo::GatherDimensionNumbers(
    offset_dims = offset_dims - 1L,
    collapsed_slice_dims = collapsed_slice_dims - 1L,
    operand_batching_dims = operand_batching_dims - 1L,
    start_indices_batching_dims = start_indices_batching_dims - 1L,
    start_index_map = start_index_map - 1L,
    index_vector_dim = index_vector_dim - 1L
  )

  one <- stablehlo::hlo_tensor(1L, dtype = dtype(start_indices), shape = shape(start_indices))
  start_indices_0based <- stablehlo::hlo_subtract(start_indices, one)

  result <- stablehlo::hlo_gather(
    operand,
    start_indices_0based,
    gather_dimension_numbers = gdn_0based,
    slice_sizes = slice_sizes,
    indices_are_sorted = indices_are_sorted
  )

  list(result)
}

p_cholesky[["stablehlo"]] <- function(operand, lower) {
  L <- stablehlo::hlo_cholesky(operand, lower = lower)
  # The non-triangular part of the output is implementation-defined.
  # Zero it out so downstream code (including reverse rules) never sees garbage.
  op_shape <- shape(operand$value_type)
  n <- op_shape[length(op_shape)]
  mat_shape <- c(n, n)
  rows <- stablehlo::hlo_iota(iota_dimension = 0L, dtype = "i32", shape = mat_shape, func = operand$func)
  cols <- stablehlo::hlo_iota(iota_dimension = 1L, dtype = "i32", shape = mat_shape, func = operand$func)
  mask <- if (lower) {
    stablehlo::hlo_compare(rows, cols, comparison_direction = "GE", compare_type = "SIGNED")
  } else {
    stablehlo::hlo_compare(rows, cols, comparison_direction = "LE", compare_type = "SIGNED")
  }
  if (length(op_shape) > 2L) {
    mask <- stablehlo::hlo_broadcast_in_dim(mask, (length(op_shape) - 2L):(length(op_shape) - 1L), op_shape)
  }
  zero <- stablehlo::hlo_tensor(0L, dtype = dtype(operand), shape = op_shape)
  list(stablehlo::hlo_select(mask, L, zero))
}

p_triangular_solve[["stablehlo"]] <- function(a, b, left_side, lower, unit_diagonal, transpose_a) {
  list(stablehlo::hlo_triangular_solve(
    a,
    b,
    left_side = left_side,
    lower = lower,
    unit_diagonal = unit_diagonal,
    transpose_a = transpose_a
  ))
}
