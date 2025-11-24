#' @include primitives.R
#' @include interpreter-jit.R

p_add[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_add(lhs, rhs))
}

p_mul[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_multiply(lhs, rhs))
}

p_sub[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_subtract(lhs, rhs))
}

p_neg[["jit"]] <- function(operand) {
  list(stablehlo::hlo_negate(operand))
}

p_div[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_divide(lhs, rhs))
}

p_pow[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_power(lhs, rhs))
}

p_broadcast_in_dim[["jit"]] <- function(operand, shape_out, broadcast_dimensions) {
  list(stablehlo::hlo_broadcast_in_dim(operand, broadcast_dimensions - 1L, shape_out))
}

p_dot_general[["jit"]] <- function(lhs, rhs, contracting_dims, batching_dims) {
  contracting_dims <- lapply(contracting_dims, \(x) x - 1L)
  batching_dims <- lapply(batching_dims, \(x) x - 1L)
  list(stablehlo::hlo_dot_general(lhs, rhs, contracting_dims, batching_dims))
}

p_transpose[["jit"]] <- function(operand, permutation) {
  list(stablehlo::hlo_transpose(operand, permutation - 1L))
}

p_reshape[["jit"]] <- function(operand, shape) {
  list(stablehlo::hlo_reshape(operand, shape))
}

p_concatenate[["jit"]] <- function(..., dimension) {
  dots <- list(...)
  list(stablehlo::hlo_concatenate(unlist(dots), dimension = dimension))
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

p_reduce_sum[["jit"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(0, dtype = dtype(operand), func = operand@func)
  }
  .jit_apply_reduce(stablehlo::hlo_add, operand, init, dims, drop)
}

p_reduce_prod[["jit"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(1, dtype = dtype(operand), func = operand@func)
  }
  .jit_apply_reduce(stablehlo::hlo_multiply, operand, init, dims, drop)
}


p_reduce_max[["jit"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_minval(dtype(operand), "cpu"))
  }
  .jit_apply_reduce(stablehlo::hlo_maximum, operand, init, dims, drop)
}

p_reduce_min[["jit"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_maxval(dtype(operand), "cpu"))
  }
  .jit_apply_reduce(stablehlo::hlo_minimum, operand, init, dims, drop)
}

p_reduce_any[["jit"]] <- function(operand, dims, drop) {
  init <- function(operand) {
    hlo_scalar(FALSE)
  }
  .jit_apply_reduce(stablehlo::hlo_or, operand, init, dims, drop)
}

p_reduce_all[["jit"]] <- function(operand, dims, drop) {
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
    stop("Unsupported dtype for compare")
  }
}

.jit_compare_bin <- function(direction) {
  function(lhs, rhs) {
    ct <- .compare_type_for(lhs)
    list(stablehlo::hlo_compare(lhs, rhs, comparison_direction = direction, compare_type = ct))
  }
}

p_eq[["jit"]] <- .jit_compare_bin("EQ")
p_ne[["jit"]] <- .jit_compare_bin("NE")
p_gt[["jit"]] <- .jit_compare_bin("GT")
p_ge[["jit"]] <- .jit_compare_bin("GE")
p_lt[["jit"]] <- .jit_compare_bin("LT")
p_le[["jit"]] <- .jit_compare_bin("LE")


# binary simple math jit rules ---------------------------------------------------

p_max[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_maximum(lhs, rhs))
}

p_min[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_minimum(lhs, rhs))
}

p_remainder[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_remainder(lhs, rhs))
}

p_and[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_and(lhs, rhs))
}

p_not[["jit"]] <- function(operand) {
  list(stablehlo::hlo_not(operand))
}

p_or[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_or(lhs, rhs))
}

p_xor[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_xor(lhs, rhs))
}

p_shift_left[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_shift_left(lhs, rhs))
}

p_shift_right_logical[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_shift_right_logical(lhs, rhs))
}

p_shift_right_arithmetic[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_shift_right_arithmetic(lhs, rhs))
}

p_atan2[["jit"]] <- function(lhs, rhs) {
  list(stablehlo::hlo_atan2(lhs, rhs))
}

# unary simple math jit rules ---------------------------------------------------

p_abs[["jit"]] <- function(operand) {
  list(stablehlo::hlo_abs(operand))
}

p_sqrt[["jit"]] <- function(operand) {
  list(stablehlo::hlo_sqrt(operand))
}

p_rsqrt[["jit"]] <- function(operand) {
  list(stablehlo::hlo_rsqrt(operand))
}

p_log[["jit"]] <- function(operand) {
  list(stablehlo::hlo_log(operand))
}

p_tanh[["jit"]] <- function(operand) {
  list(stablehlo::hlo_tanh(operand))
}

p_tan[["jit"]] <- function(operand) {
  list(stablehlo::hlo_tan(operand))
}

p_sine[["jit"]] <- function(operand) {
  list(stablehlo::hlo_sine(operand))
}

p_cosine[["jit"]] <- function(operand) {
  list(stablehlo::hlo_cosine(operand))
}

p_floor[["jit"]] <- function(operand) {
  list(stablehlo::hlo_floor(operand))
}

p_ceil[["jit"]] <- function(operand) {
  list(stablehlo::hlo_ceil(operand))
}

p_sign[["jit"]] <- function(operand) {
  list(stablehlo::hlo_sign(operand))
}

p_exp[["jit"]] <- function(operand) {
  list(stablehlo::hlo_exponential(operand))
}

p_round[["jit"]] <- function(operand, method) {
  switch(
    method,
    afz = list(stablehlo::hlo_round_nearest_afz(operand)),
    nearest_even = list(stablehlo::hlo_round_nearest_even(operand)),
    cli_abort("invalid method: {method}")
  )
}

p_convert[["jit"]] <- function(operand, dtype) {
  list(stablehlo::hlo_convert(operand, dtype))
}

# control flow jit rules --------------------------------------------------------

p_select[["jit"]] <- function(pred, true_value, false_value) {
  list(stablehlo::hlo_select(pred, true_value, false_value))
}

# RNG jit rules --------------------------------------------------------

p_rng_bit_generator[["jit"]] <- function(initial_state, rng_algorithm, dtype, shape_out) {
  stablehlo::hlo_rng_bit_generator(initial_state, rng_algorithm, dtype, shape_out)
}
