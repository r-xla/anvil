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
