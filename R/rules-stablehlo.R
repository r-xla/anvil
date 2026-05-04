#' @include primitives.R

# TODO: Here we don't have to re-do the type inference again, because it was already done.

prim_fill[["stablehlo"]] <- function(value, shape, dtype, ambiguous) {
  # ambiguity only relevant for type promotion, but when we lower
  # there is no type promotion, so it has no effect
  list(stablehlo::hlo_tensor(value, shape = shape, dtype = dtype))
}

prim_add[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_add(lhs, rhs))
}

prim_mul[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_multiply(lhs, rhs))
}

prim_sub[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_subtract(lhs, rhs))
}

prim_negate[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_negate(operand))
}

prim_div[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_divide(lhs, rhs))
}

prim_pow[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_power(lhs, rhs))
}

prim_broadcast_in_dim[["stablehlo"]] <- function(operand, shape, broadcast_dimensions) {
  list(stablehlo::hlo_broadcast_in_dim(operand, broadcast_dimensions - 1L, shape))
}

prim_dot_general[["stablehlo"]] <- function(lhs, rhs, contracting_dims, batching_dims) {
  contracting_dims <- lapply(contracting_dims, \(x) x - 1L)
  batching_dims <- lapply(batching_dims, \(x) x - 1L)
  list(stablehlo::hlo_dot_general(lhs, rhs, contracting_dims, batching_dims))
}

prim_transpose[["stablehlo"]] <- function(operand, permutation) {
  list(stablehlo::hlo_transpose(operand, permutation - 1L))
}

prim_reshape[["stablehlo"]] <- function(operand, shape) {
  list(stablehlo::hlo_reshape(operand, shape))
}

prim_concatenate[["stablehlo"]] <- function(..., dimension) {
  list(stablehlo::hlo_concatenate(..., dimension = dimension - 1L))
}

prim_static_slice[["stablehlo"]] <- function(operand, start_indices, limit_indices, strides) {
  # we use 1:n, which includes n, but this translates to 0:n in stablehlo
  list(stablehlo::hlo_slice(operand, start_indices - 1L, limit_indices, strides))
}

prim_dynamic_slice[["stablehlo"]] <- function(operand, ..., slice_sizes) {
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

prim_dynamic_update_slice[["stablehlo"]] <- function(operand, update, ...) {
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

prim_reduce_sum[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(0, dtype = dtype(operand), func = operand$func)
  }
  .stablehlo_apply_reduce(stablehlo::hlo_add, operand, init, dims, drop)
}

prim_reduce_prod[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(1, dtype = dtype(operand), func = operand$func)
  }
  .stablehlo_apply_reduce(stablehlo::hlo_multiply, operand, init, dims, drop)
}


prim_reduce_max[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_minval(dtype(operand), "cpu"))
  }
  .stablehlo_apply_reduce(stablehlo::hlo_maximum, operand, init, dims, drop)
}

prim_reduce_min[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_maxval(dtype(operand), "cpu"))
  }
  .stablehlo_apply_reduce(stablehlo::hlo_minimum, operand, init, dims, drop)
}

prim_reduce_any[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(FALSE)
  }
  .stablehlo_apply_reduce(stablehlo::hlo_or, operand, init, dims, drop)
}

prim_reduce_all[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(TRUE)
  }
  .stablehlo_apply_reduce(stablehlo::hlo_and, operand, init, dims, drop)
}

.stablehlo_apply_cum <- function(reductor, operand, init, dim) {
  shp <- shape(operand)
  rank <- length(shp)
  s_d <- shp[[dim]]
  window_dimensions <- rep(1L, rank)
  window_dimensions[[dim]] <- s_d
  ones <- rep(1L, rank)
  padding <- matrix(0L, nrow = rank, ncol = 2L)
  padding[dim, 1L] <- s_d - 1L

  local_func("")
  dt <- as.character(operand$value_type$type$dtype)
  body <- hlo_return(reductor(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))

  list(stablehlo::hlo_reduce_window(
    inputs = operand,
    init_values = init(operand),
    window_dimensions = window_dimensions,
    window_strides = ones,
    base_dilations = ones,
    window_dilations = ones,
    padding = padding,
    body = body
  ))
}

prim_cumsum[["stablehlo"]] <- function(operand, dim) {
  init <- function(operand) {
    hlo_scalar(0, dtype = dtype(operand), func = operand$func)
  }
  .stablehlo_apply_cum(stablehlo::hlo_add, operand, init, dim)
}

prim_cumprod[["stablehlo"]] <- function(operand, dim) {
  init <- function(operand) {
    hlo_scalar(1, dtype = dtype(operand), func = operand$func)
  }
  .stablehlo_apply_cum(stablehlo::hlo_multiply, operand, init, dim)
}

.stablehlo_apply_cum_extreme <- function(operand, dim, is_max) {
  shp <- shape(operand)
  rank <- length(shp)
  s_d <- shp[[dim]]
  window_dimensions <- rep(1L, rank)
  window_dimensions[[dim]] <- s_d
  ones <- rep(1L, rank)
  padding <- matrix(0L, nrow = rank, ncol = 2L)
  padding[dim, 1L] <- s_d - 1L

  v_dtype <- as.character(operand$value_type$type$dtype)
  iota <- stablehlo::hlo_iota(iota_dimension = dim - 1L, dtype = "i32", shape = shp)

  init_v_fn <- if (is_max) nv_minval else nv_maxval
  init_v <- hlo_scalar(init_v_fn(v_dtype, "cpu"))
  init_i <- hlo_scalar(0L, dtype = "i32", func = operand$func)

  cmp <- if (is_max) prim_gt else prim_lt
  # Pick the side with the strictly better value; on ties, the smaller index
  # wins (first-occurrence tiebreak). The op is associative + commutative.
  reductor <- function(lv, li, rv, ri) {
    lhs_wins <- prim_or(cmp(lv, rv), prim_and(prim_eq(lv, rv), prim_lt(li, ri)))
    list(nv_ifelse(lhs_wins, lv, rv), nv_ifelse(lhs_wins, li, ri))
  }
  body <- .r_reductor_to_hlo_func(
    reductor,
    list(
      lv = nv_aval(v_dtype, integer()),
      li = nv_aval("i32", integer()),
      rv = nv_aval(v_dtype, integer()),
      ri = nv_aval("i32", integer())
    )
  )

  out <- stablehlo::hlo_reduce_window(
    inputs = list(operand, iota),
    init_values = list(init_v, init_i),
    window_dimensions = window_dimensions,
    window_strides = ones,
    base_dilations = ones,
    window_dilations = ones,
    padding = padding,
    body = body
  )
  values <- out[[1L]]
  indices_0 <- out[[2L]]
  one <- stablehlo::hlo_scalar(1L, dtype = "i32", func = indices_0$func)
  one_bc <- stablehlo::hlo_broadcast_in_dim(one, integer(0), shape(indices_0$value_type))
  list(values, stablehlo::hlo_add(indices_0, one_bc))
}

prim_cummax[["stablehlo"]] <- function(operand, dim) {
  .stablehlo_apply_cum_extreme(operand, dim, is_max = TRUE)
}

prim_cummin[["stablehlo"]] <- function(operand, dim) {
  .stablehlo_apply_cum_extreme(operand, dim, is_max = FALSE)
}

prim_reduce[["stablehlo"]] <- function(operand, init, dims, drop, reductor_graph, .env) {
  red_func <- stablehlo(reductor_graph)[[1L]]
  out <- stablehlo::hlo_reduce(
    inputs = list(operand),
    init_values = list(init),
    dimensions = dims - 1L,
    body = red_func
  )
  if (drop) {
    return(list(out))
  }
  shape_out <- shape(operand$value_type)
  shape_out[dims] <- 1L
  list(stablehlo::hlo_reshape(out, shape_out))
}

.r_reductor_to_hlo_func <- function(fn, dummy_args) {
  graph <- trace_fn(fn, dummy_args, desc = local_descriptor())
  stablehlo(graph, constants_as_inputs = FALSE)[[1L]]
}

.stablehlo_arg_extreme <- function(operand, dim, drop, direction, init_v_fn) {
  shp <- shape(operand$value_type)
  v_dtype <- operand$value_type$type$dtype
  iota <- stablehlo::hlo_iota(iota_dimension = dim - 1L, dtype = "i32", shape = shp)
  init_v <- hlo_scalar(init_v_fn(v_dtype, "cpu"))
  init_i <- hlo_scalar(0L, dtype = "i32", func = operand$func)

  # Reductor: pick lhs unless rhs is strictly better.
  # For ties, we pick li (which we know is < ri, although reductor is applied in random order and
  # thus assumed to be associative, it does not have to be commutative)
  cmp <- if (direction == "GT") prim_gt else prim_lt
  reductor <- function(lv, li, rv, ri) {
    rhs_better <- cmp(rv, lv)
    list(nv_ifelse(rhs_better, rv, lv), nv_ifelse(rhs_better, ri, li))
  }
  body <- .r_reductor_to_hlo_func(
    reductor,
    list(
      lv = nv_aval(v_dtype, integer()),
      li = nv_aval("i32", integer()),
      rv = nv_aval(v_dtype, integer()),
      ri = nv_aval("i32", integer())
    )
  )

  out <- stablehlo::hlo_reduce(
    inputs = list(operand, iota),
    init_values = list(init_v, init_i),
    dimensions = dim - 1L,
    body = body
  )
  # convert to 1-based
  result <- out[[2L]]
  one <- stablehlo::hlo_scalar(1L, dtype = "i32", func = result$func)
  one_bc <- stablehlo::hlo_broadcast_in_dim(one, integer(0), shape(result$value_type))
  result <- stablehlo::hlo_add(result, one_bc)
  if (drop) {
    return(list(result))
  }
  shape_out <- shp
  shape_out[dim] <- 1L
  list(stablehlo::hlo_reshape(result, shape_out))
}

prim_argmax[["stablehlo"]] <- function(operand, dim, drop) {
  .stablehlo_arg_extreme(operand, dim, drop, direction = "GT", init_v_fn = nv_minval)
}

prim_argmin[["stablehlo"]] <- function(operand, dim, drop) {
  .stablehlo_arg_extreme(operand, dim, drop, direction = "LT", init_v_fn = nv_maxval)
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

prim_eq[["stablehlo"]] <- .stablehlo_compare_bin("EQ")
prim_ne[["stablehlo"]] <- .stablehlo_compare_bin("NE")
prim_gt[["stablehlo"]] <- .stablehlo_compare_bin("GT")
prim_ge[["stablehlo"]] <- .stablehlo_compare_bin("GE")
prim_lt[["stablehlo"]] <- .stablehlo_compare_bin("LT")
prim_le[["stablehlo"]] <- .stablehlo_compare_bin("LE")


# binary simple math jit rules ---------------------------------------------------

prim_max[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_maximum(lhs, rhs))
}

prim_min[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_minimum(lhs, rhs))
}

prim_remainder[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_remainder(lhs, rhs))
}

prim_and[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_and(lhs, rhs))
}

prim_not[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_not(operand))
}

prim_or[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_or(lhs, rhs))
}

prim_xor[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_xor(lhs, rhs))
}

prim_shift_left[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_shift_left(lhs, rhs))
}

prim_shift_right_logical[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_shift_right_logical(lhs, rhs))
}

prim_shift_right_arithmetic[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_shift_right_arithmetic(lhs, rhs))
}

prim_atan2[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_atan2(lhs, rhs))
}

prim_bitcast_convert[["stablehlo"]] <- function(operand, dtype) {
  list(stablehlo::hlo_bitcast_convert(operand, dtype))
}

# unary simple math jit rules ---------------------------------------------------

prim_abs[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_abs(operand))
}

prim_sqrt[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_sqrt(operand))
}

prim_rsqrt[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_rsqrt(operand))
}

prim_log[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_log(operand))
}

prim_tanh[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_tanh(operand))
}

prim_tan[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_tan(operand))
}

prim_sine[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_sine(operand))
}

prim_cosine[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_cosine(operand))
}

prim_floor[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_floor(operand))
}

prim_ceil[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_ceil(operand))
}

prim_sign[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_sign(operand))
}

prim_exp[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_exponential(operand))
}

prim_expm1[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_exponential_minus_one(operand))
}

prim_log1p[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_log_plus_one(operand))
}

prim_cbrt[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_cbrt(operand))
}

prim_logistic[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_logistic(operand))
}

prim_is_finite[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_is_finite(operand))
}

prim_popcnt[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_popcnt(operand))
}

prim_clamp[["stablehlo"]] <- function(min_val, operand, max_val) {
  list(stablehlo::hlo_clamp(min_val, operand, max_val))
}

prim_reverse[["stablehlo"]] <- function(operand, dims) {
  list(stablehlo::hlo_reverse(operand, dims - 1L))
}

prim_iota[["stablehlo"]] <- function(dim, dtype, shape, start, ambiguous) {
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

prim_pad[["stablehlo"]] <- function(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding) {
  list(stablehlo::hlo_pad(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding))
}

prim_round[["stablehlo"]] <- function(operand, method) {
  switch(
    method,
    afz = list(stablehlo::hlo_round_nearest_afz(operand)),
    nearest_even = list(stablehlo::hlo_round_nearest_even(operand)),
    cli_abort("invalid method: {method}")
  )
}

prim_convert[["stablehlo"]] <- function(operand, dtype, ambiguous) {
  list(stablehlo::hlo_convert(operand, dtype))
}


prim_ifelse[["stablehlo"]] <- function(pred, true_value, false_value) {
  list(stablehlo::hlo_select(pred, true_value, false_value))
}

# RNG jit rules --------------------------------------------------------

prim_rng_bit_generator[["stablehlo"]] <- function(initial_state, rng_algorithm, dtype, shape) {
  stablehlo::hlo_rng_bit_generator(initial_state, rng_algorithm, dtype, shape)
}

prim_print[["stablehlo"]] <- function(operand, footer) {
  backend_config <- stablehlo::CustomOpBackendConfig(list(
    stablehlo::StringAttr(name = "print_header", value = "AnvlArray"),
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

prim_if[["stablehlo"]] <- function(pred, true_graph, false_graph, .env) {
  true_func <- stablehlo(true_graph, constants_as_inputs = FALSE, env = .env)[[1L]]
  false_func <- stablehlo(false_graph, constants_as_inputs = FALSE, env = .env)[[1L]]
  stablehlo::hlo_if(pred, true_func, false_func, simplify = FALSE)
}

prim_while[["stablehlo"]] <- function(..., cond_graph, body_graph, .env) {
  body_func <- stablehlo(body_graph, constants_as_inputs = FALSE, env = .env)[[1L]]
  cond_func <- stablehlo(cond_graph, constants_as_inputs = FALSE, env = .env)[[1L]]
  stablehlo::hlo_while(..., cond = cond_func, body = body_func, simplify = FALSE)
}

prim_sort[["stablehlo"]] <- function(..., dim, descending, is_stable) {
  ops <- list(...)
  cmp <- if (descending) `>` else `<`
  # Comparator takes 2*N scalars (one pair per operand) but only ranks by
  # the first key pair; the remaining operands ride along.
  comparator <- function(...) {
    args <- list(...)
    cmp(args[[1L]], args[[2L]])
  }
  dummy_args <- unlist(
    lapply(ops, function(op) {
      dt <- op$value_type$type$dtype
      list(
        nv_aval(dtype = dt, shape = integer()),
        nv_aval(dtype = dt, shape = integer())
      )
    }),
    recursive = FALSE
  )
  cmp_func <- .r_reductor_to_hlo_func(comparator, dummy_args)
  stablehlo::hlo_sort(
    ...,
    dimension = dim - 1L,
    is_stable = is_stable,
    comparator = cmp_func
  )
}

prim_top_k[["stablehlo"]] <- function(operand, k) {
  out <- stablehlo::hlo_top_k(operand, k = k)
  values <- out[[1L]]
  indices <- out[[2L]]

  # to 1-based
  one <- stablehlo::hlo_scalar(1L, dtype = "i32", func = indices$func)
  one_bc <- stablehlo::hlo_broadcast_in_dim(one, integer(0), shape(indices$value_type))
  indices <- stablehlo::hlo_add(indices, one_bc)

  list(values, indices)
}

prim_scatter[["stablehlo"]] <- function(
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

prim_gather[["stablehlo"]] <- function(
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

prim_cholesky[["stablehlo"]] <- function(operand, lower) {
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

prim_triangular_solve[["stablehlo"]] <- function(a, b, left_side, lower, unit_diagonal, transpose_a) {
  list(stablehlo::hlo_triangular_solve(
    a,
    b,
    left_side = left_side,
    lower = lower,
    unit_diagonal = unit_diagonal,
    transpose_a = transpose_a
  ))
}
