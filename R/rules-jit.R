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

p_reduce_sum[["jit"]] <- function(operand, dims, drop) {
  local_func("")

  # TODO: Simplify?
  dt <- as.character(operand@value_type@type@dtype)
  f <- hlo_return(stablehlo::hlo_add(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))
  init <- hlo_scalar(0, dtype = dt, func = operand@func)
  out <- stablehlo::hlo_reduce(list(operand), init, dims - 1L, f)

  if (drop) {
    return(list(out))
  }

  shape_out <- shape(operand@value_type)
  shape_out[dims] <- 1L
  list(stablehlo::hlo_reshape(out, shape_out))
}

p_reduce_prod[["jit"]] <- function(operand, dims, drop) {
  local_func("")
  dt <- as.character(operand@value_type@type@dtype)
  f <- hlo_return(stablehlo::hlo_multiply(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))
  init <- hlo_scalar(1, dtype = dt, func = operand@func)
  out <- stablehlo::hlo_reduce(list(operand), init, dims - 1L, f)

  if (drop) {
    return(list(out))
  }

  shape_out <- shape(operand@value_type)
  shape_out[dims] <- 1L
  list(stablehlo::hlo_reshape(out, shape_out))
}

p_reduce_max[["jit"]] <- function(operand, dims, drop) {
  local_func("")
  dt <- as.character(operand@value_type@type@dtype)
  f <- hlo_return(stablehlo::hlo_maximum(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))
  init <- if (startsWith(dt, "f")) {
    hlo_scalar(-Inf, dtype = dt, func = operand@func)
  } else if (startsWith(dt, "ui")) {
    hlo_scalar(0L, dtype = dt, func = operand@func)
  } else if (dt == "i1") {
    hlo_scalar(FALSE, func = operand@func)
  } else {
    # signed integer
    bits <- as.integer(sub("^i", "", dt))
    hlo_scalar(as.double(-(2^(bits - 1))), dtype = dt, func = operand@func)
  }
  out <- stablehlo::hlo_reduce(list(operand), init, dims - 1L, f)

  if (drop) {
    return(list(out))
  }

  shape_out <- shape(operand@value_type)
  shape_out[dims] <- 1L
  list(stablehlo::hlo_reshape(out, shape_out))
}

p_reduce_min[["jit"]] <- function(operand, dims, drop) {
  local_func("")
  dt <- as.character(operand@value_type@type@dtype)
  f <- hlo_return(stablehlo::hlo_minimum(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))
  init <- if (startsWith(dt, "f")) {
    hlo_scalar(Inf, dtype = dt, func = operand@func)
  } else if (startsWith(dt, "ui")) {
    bits <- as.integer(sub("^ui", "", dt))
    hlo_scalar(as.double(2^bits - 1), dtype = dt, func = operand@func)
  } else if (dt == "i1") {
    hlo_scalar(TRUE, func = operand@func)
  } else {
    # signed integer
    bits <- as.integer(sub("^i", "", dt))
    hlo_scalar(as.double(2^(bits - 1) - 1), dtype = dt, func = operand@func)
  }
  out <- stablehlo::hlo_reduce(list(operand), init, dims - 1L, f)

  if (drop) {
    return(list(out))
  }

  shape_out <- shape(operand@value_type)
  shape_out[dims] <- 1L
  list(stablehlo::hlo_reshape(out, shape_out))
}

p_reduce_any[["jit"]] <- function(operand, dims, drop) {
  local_func("")
  dt <- as.character(operand@value_type@type@dtype)
  f <- hlo_return(stablehlo::hlo_or(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))
  init <- hlo_scalar(FALSE, func = operand@func)
  out <- stablehlo::hlo_reduce(list(operand), init, dims - 1L, f)

  if (drop) {
    return(list(out))
  }

  shape_out <- shape(operand@value_type)
  shape_out[dims] <- 1L
  list(stablehlo::hlo_reshape(out, shape_out))
}

p_reduce_all[["jit"]] <- function(operand, dims, drop) {
  local_func("")
  dt <- dtype(operand)
  f <- hlo_return(stablehlo::hlo_and(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))
  init <- hlo_scalar(TRUE, func = operand@func)
  out <- stablehlo::hlo_reduce(list(operand), init, dims - 1L, f)

  if (drop) {
    return(list(out))
  }

  shape_out <- shape(operand@value_type)
  shape_out[dims] <- 1L
  list(stablehlo::hlo_reshape(out, shape_out))
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

# additional simple binary jit rules ------------------------------------------

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
  switch(method,
    afz = list(stablehlo::hlo_round_nearest_afz(operand)),
    nearest_even = list(stablehlo::hlo_round_nearest_even(operand)),
    cli_abort("invalid method: {method}")
  )
}

# control flow jit rules --------------------------------------------------------

p_select[["jit"]] <- function(pred, true_value, false_value) {
  list(stablehlo::hlo_select(pred, true_value, false_value))
}
