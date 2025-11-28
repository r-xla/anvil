#' @include primitives.R

# TODO: Here we don't have to re-do the type inference again, because it was already done.

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

.jit_apply_reduce <- function(reductor, operand, init, dims, drop) {
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
  .jit_apply_reduce(stablehlo::hlo_add, operand, init, dims, drop)
}

p_reduce_prod[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(1, dtype = dtype(operand), func = operand@func)
  }
  .jit_apply_reduce(stablehlo::hlo_multiply, operand, init, dims, drop)
}


p_reduce_max[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_minval(dtype(operand), "cpu"))
  }
  .jit_apply_reduce(stablehlo::hlo_maximum, operand, init, dims, drop)
}

p_reduce_min[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_maxval(dtype(operand), "cpu"))
  }
  .jit_apply_reduce(stablehlo::hlo_minimum, operand, init, dims, drop)
}

p_reduce_any[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(FALSE)
  }
  .jit_apply_reduce(stablehlo::hlo_or, operand, init, dims, drop)
}

p_reduce_all[["stablehlo"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(TRUE)
  }
  .jit_apply_reduce(stablehlo::hlo_and, operand, init, dims, drop)
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

.jit_compare_bin <- function(direction) {
  function(lhs, rhs) {
    ct <- .compare_type_for(lhs)
    list(stablehlo::hlo_compare(lhs, rhs, comparison_direction = direction, compare_type = ct))
  }
}

p_eq[["stablehlo"]] <- .jit_compare_bin("EQ")
p_ne[["stablehlo"]] <- .jit_compare_bin("NE")
p_gt[["stablehlo"]] <- .jit_compare_bin("GT")
p_ge[["stablehlo"]] <- .jit_compare_bin("GE")
p_lt[["stablehlo"]] <- .jit_compare_bin("LT")
p_le[["stablehlo"]] <- .jit_compare_bin("LE")


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

p_round[["stablehlo"]] <- function(operand, method) {
  switch(
    method,
    afz = list(stablehlo::hlo_round_nearest_afz(operand)),
    nearest_even = list(stablehlo::hlo_round_nearest_even(operand)),
    cli_abort("invalid method: {method}")
  )
}

p_convert[["stablehlo"]] <- function(operand, dtype) {
  list(stablehlo::hlo_convert(operand, dtype))
}


p_select[["stablehlo"]] <- function(pred, true_value, false_value) {
  list(stablehlo::hlo_select(pred, true_value, false_value))
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
