#' @include primitives.R
#' @include interpreter-jit.R

register_jit_rule(p_add, function(lhs, rhs) {
  list(stablehlo::hlo_add(lhs, rhs))
})

register_jit_rule(p_mul, function(lhs, rhs) {
  list(stablehlo::hlo_multiply(lhs, rhs))
})

register_jit_rule(p_sub, function(lhs, rhs) {
  list(stablehlo::hlo_subtract(lhs, rhs))
})

register_jit_rule(p_neg, function(operand) {
  list(stablehlo::hlo_negate(operand))
})

register_jit_rule(p_div, function(lhs, rhs) {
  list(stablehlo::hlo_divide(lhs, rhs))
})

register_jit_rule(p_pow, function(lhs, rhs) {
  list(stablehlo::hlo_power(lhs, rhs))
})

register_jit_rule(
  p_broadcast_in_dim,
  function(operand, shape_out, broadcast_dimensions) {
    list(stablehlo::hlo_broadcast_in_dim(operand, broadcast_dimensions - 1L, shape_out))
  }
)

register_jit_rule(
  p_dot_general,
  function(lhs, rhs, contracting_dims, batching_dims = NULL) {
    contracting_dims <- lapply(contracting_dims, \(x) x - 1L)
    batching_dims <- lapply(batching_dims, \(x) x - 1L)
    list(stablehlo::hlo_dot_general(lhs, rhs, contracting_dims, batching_dims))
  }
)

register_jit_rule(p_transpose, function(operand, permutation) {
  list(stablehlo::hlo_transpose(operand, permutation - 1L))
})

register_jit_rule(p_reshape, function(operand, shape) {
  list(stablehlo::hlo_reshape(operand, shape))
})

register_jit_rule(p_reduce_sum, function(operand, dims, drop) {
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
})
