#' @include primitives.R
#' @include interpreter-jit.R

register_jit_rule(p_add, function(lhs, rhs) {
  list(hlo_add(lhs, rhs))
})

register_jit_rule(p_mul, function(lhs, rhs) {
  list(hlo_multiply(lhs, rhs))
})

register_jit_rule(p_sub, function(lhs, rhs) {
  list(hlo_subtract(lhs, rhs))
})

register_jit_rule(p_neg, function(operand) {
  list(hlo_negate(operand))
})

register_jit_rule(
  p_broadcast_in_dim,
  function(operand, shape_out, broadcast_dimensions) {
    list(hlo_broadcast_in_dim(operand, broadcast_dimensions, shape_out))
  }
)

register_jit_rule(
  p_dot_general,
  function(lhs, rhs, contracting_dims, batching_dims = NULL) {
    list(hlo_dot_general(lhs, rhs, contracting_dims, batching_dims))
  }
)

register_jit_rule(p_transpose, function(operand, permutation) {
  list(hlo_transpose(operand, permutation))
})
