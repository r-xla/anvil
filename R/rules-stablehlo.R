#' @include primitives.R

# TODO: Here we don't have to re-do the type inference again, because it was already done.

p_fill[["stablehlo"]] <- function(value, shape, dtype) {
  if (is.null(shape)) {
    shape <- integer()
  }
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

p_neg[["stablehlo"]] <- function(operand) {
  list(stablehlo::hlo_negate(operand))
}

p_div[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_divide(lhs, rhs))
}

p_pow[["stablehlo"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_power(lhs, rhs))
}

p_broadcast_in_dim[["stablehlo"]] <- function(operand, shape_out, broadcast_dimensions) {
  list(stablehlo::hlo_broadcast_in_dim(operand, broadcast_dimensions - 1L, shape_out))
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
  dim_arg <- stablehlo_concat_dim(dimension)
  list(stablehlo::hlo_concatenate(..., dimension = dim_arg))
}

p_slice[["stablehlo"]] <- function(operand, start_indices, limit_indices, strides) {
  # we use 1:n, which includes n, but this translates to 0:n in stablehlo
  list(stablehlo::hlo_slice(operand, start_indices - 1L, limit_indices, strides))
}

.stablehlo_apply_reduce <- function(reductor, operand, init, dims, drop) {
  local_func("")
  dt <- as.character(operand@value_type@type@dtype)
  f <- hlo_return(reductor(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))
  out <- stablehlo::hlo_reduce(list(operand), init(operand), dims - 1L, f)

  if (drop) {
    return(list(out))
  }

  shape_out <- shape(operand@value_type)
  shape_out[dims] <- 1L
  list(stablehlo::hlo_reshape(out, shape_out))
}

p_reduce_sum[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(0, dtype = dtype(operand), func = operand@func)
  }
  .stablehlo_apply_reduce(stablehlo::hlo_add, operand, init, dims, drop)
}

p_reduce_prod[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(1, dtype = dtype(operand), func = operand@func)
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

# comparison jit rules ----------------------------------------------------------

.compare_type_for <- function(vt) {
  dt <- vt@value_type@type@dtype
  if (inherits(dt, stablehlo::FloatType)) {
    "FLOAT"
  } else if (inherits(dt, stablehlo::IntegerType)) {
    "SIGNED"
  } else if (inherits(dt, stablehlo::UnsignedType)) {
    "UNSIGNED"
  } else if (inherits(dt, stablehlo::BooleanType)) {
    # StableHLO uses SIGNED for i1 compares
    "SIGNED"
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
  shape_out <- shape(operand@value_type)
  maybe_broadcast <- function(x) {
    x_shape <- shape(x@value_type)
    if (!length(x_shape) && length(shape_out)) {
      return(stablehlo::hlo_broadcast_in_dim(x, integer(), shape_out))
    }
    x
  }
  min_val <- maybe_broadcast(min_val)
  max_val <- maybe_broadcast(max_val)
  list(stablehlo::hlo_clamp(min_val, operand, max_val))
}

p_reverse[["stablehlo"]] <- function(operand, dims) {
  dims <- as.integer(dims)
  dims_attr <- local({
    func <- stablehlo::local_func("")
    stablehlo:::impl_hlo_constant(dims - 1L, dtype = "i64", func = func, shape = length(dims))
  })
  list(stablehlo:::hlo_reverse_impl(values = list(operand = operand), attrs = list(dimensions = dims_attr)))
}

p_iota[["stablehlo"]] <- function(dim, dtype, shape) {
  ns <- asNamespace("stablehlo")
  hlo_iota <- get0("hlo_iota", envir = ns, inherits = FALSE)
  if (is.function(hlo_iota)) {
    dim_arg <- stablehlo_iota_dim(dim)
    return(list(hlo_iota(iota_dimension = dim_arg, dtype = dtype, shape = shape)))
  }
  shape <- as.integer(shape)
  if (length(shape) && (dim < 1L || dim > length(shape))) {
    cli_abort("{.arg dim} must be between 1 and {length(shape)}")
  }
  if (!length(shape)) {
    arr <- array(0, dim = integer())
  } else if (prod(shape) == 0L) {
    arr <- array(integer(), dim = shape)
  } else {
    idx <- arrayInd(seq_len(prod(shape)), .dim = shape)
    arr <- array(idx[, dim] - 1L, dim = shape)
  }
  list(stablehlo::hlo_tensor(arr, shape = shape, dtype = dtype))
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

p_rng_bit_generator[["stablehlo"]] <- function(initial_state, rng_algorithm, dtype, shape_out) {
  stablehlo::hlo_rng_bit_generator(initial_state, rng_algorithm, dtype, shape_out)
}

stablehlo_has_custom_call <- function() {
  ns <- asNamespace("stablehlo")
  exists("hlo_custom_call", envir = ns, inherits = FALSE) &&
    exists("CustomOpBackendConfig", envir = ns, inherits = FALSE) &&
    exists("StringAttr", envir = ns, inherits = FALSE)
}

p_print[["stablehlo"]] <- function(operand) {
  if (!stablehlo_has_custom_call()) {
    return(list(operand))
  }
  backend_config <- stablehlo::CustomOpBackendConfig(list(
    stablehlo::StringAttr(name = "print_header", value = "AnvilTensor")
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
