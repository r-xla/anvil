# The graph rules are very simple, we do no type inference, but just build the graph and
# make closed over constants explicit, which is e.g. important for the pullback rule.

graph_rule_binary <- function(lhs, rhs) {
  stablehlo::infer_types_generic_biv(lhs, rhs)@items
}

# Helper for unary operations - infer output type from input type
graph_rule_unary <- function(operand) {
  stablehlo::infer_types_generic_uni(operand)@items
}

graph_rule_binary_boolean <- function(lhs, rhs) {
  stablehlo::infer_types_boolean_biv(lhs, rhs)@items
}

graph_rule_unary_boolean <- function(operand) {
  stablehlo::infer_types_boolean_uni(operand)@items
}

#' @importFrom stablehlo infer_types_generic_biv infer_types_generic_uni
#' @importFrom stablehlo infer_types_boolean_biv infer_types_boolean_uni
#' @importFrom stablehlo infer_types_compare infer_types_transpose infer_types_reshape
#' @importFrom stablehlo infer_types_broadcast_in_dim infer_types_convert
#' @importFrom stablehlo infer_types_dot_general infer_types_select
p_add[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary(lhs, rhs)
}

p_mul[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary(lhs, rhs)
}

p_sub[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary(lhs, rhs)
}

p_neg[["graph"]] <- function(operand) {
  graph_rule_unary(operand)
}

p_div[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary(lhs, rhs)
}

p_pow[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary(lhs, rhs)
}

# Binary operations with type inference

p_max[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary(lhs, rhs)
}

p_min[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary(lhs, rhs)
}

p_remainder[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary(lhs, rhs)
}

p_and[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary_boolean(lhs, rhs)
}

p_or[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary_boolean(lhs, rhs)
}

p_xor[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary_boolean(lhs, rhs)
}

p_shift_left[["graph"]] <- function(lhs, rhs) {
  stablehlo::infer_types_shift_left(lhs, rhs)@items
}

p_shift_right_logical[["graph"]] <- function(lhs, rhs) {
  stablehlo::infer_types_shift_right_logical(lhs, rhs)@items
}

p_shift_right_arithmetic[["graph"]] <- function(lhs, rhs) {
  stablehlo::infer_types_shift_right_arithmetic(lhs, rhs)@items
}

p_atan2[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary(lhs, rhs)
}

# Comparison operations (all return boolean type)

p_eq[["graph"]] <- function(lhs, rhs) {
  stablehlo::infer_types_compare(lhs, rhs, "EQ", "FLOAT")@items
}

p_ne[["graph"]] <- function(lhs, rhs) {
  stablehlo::infer_types_compare(lhs, rhs, "NE", "FLOAT")@items
}

p_gt[["graph"]] <- function(lhs, rhs) {
  stablehlo::infer_types_compare(lhs, rhs, "GT", "FLOAT")@items
}

p_ge[["graph"]] <- function(lhs, rhs) {
  stablehlo::infer_types_compare(lhs, rhs, "GE", "FLOAT")@items
}

p_lt[["graph"]] <- function(lhs, rhs) {
  stablehlo::infer_types_compare(lhs, rhs, "LT", "FLOAT")@items
}

p_le[["graph"]] <- function(lhs, rhs) {
  stablehlo::infer_types_compare(lhs, rhs, "LE", "FLOAT")@items
}

# Unary operations

p_abs[["graph"]] <- function(operand) {
  graph_rule_unary(operand)
}

p_sqrt[["graph"]] <- function(operand) {
  graph_rule_unary(operand)
}

p_rsqrt[["graph"]] <- function(operand) {
  graph_rule_unary(operand)
}

p_log[["graph"]] <- function(operand) {
  graph_rule_unary(operand)
}

p_tanh[["graph"]] <- function(operand) {
  graph_rule_unary(operand)
}

p_tan[["graph"]] <- function(operand) {
  graph_rule_unary(operand)
}

p_floor[["graph"]] <- function(operand) {
  graph_rule_unary(operand)
}

p_ceil[["graph"]] <- function(operand) {
  graph_rule_unary(operand)
}

p_sign[["graph"]] <- function(operand) {
  graph_rule_unary(operand)
}

p_exp[["graph"]] <- function(operand) {
  graph_rule_unary(operand)
}

p_not[["graph"]] <- function(operand) {
  graph_rule_unary_boolean(operand)
}

# Round operation with method parameter

p_round[["graph"]] <- function(operand, method) {
  graph_rule_unary(operand)
}

# Type conversion

p_convert[["graph"]] <- function(operand, dtype) {
  stablehlo::infer_types_convert(operand, dtype)@items
}

# Shape-manipulating operations

p_broadcast_in_dim[["graph"]] <- function(operand, shape_out, broadcast_dimensions) {
  bd_attr <- stablehlo::r_to_constant(
    as.integer(broadcast_dimensions - 1L),
    dtype = "i64",
    shape = length(broadcast_dimensions)
  )
  stablehlo::infer_types_broadcast_in_dim(
    operand,
    broadcast_dimensions = bd_attr,
    shape_out = shape_out
  )@items
}

p_reshape[["graph"]] <- function(operand, shape) {
  stablehlo::infer_types_reshape(operand, shape_out = shape)@items
}

p_transpose[["graph"]] <- function(operand, permutation) {
  perm_attr <- stablehlo::r_to_constant(
    as.integer(permutation - 1L),
    dtype = "i64",
    shape = length(permutation)
  )
  stablehlo::infer_types_transpose(operand, permutation = perm_attr)@items
}

# Matrix multiplication

p_dot_general[["graph"]] <- function(lhs, rhs, contracting_dims, batching_dims) {
  ddn <- stablehlo::DotDimensionNumbers(
    contracting_dims = lapply(contracting_dims, \(x) x - 1L),
    batching_dims = lapply(batching_dims, \(x) x - 1L)
  )
  stablehlo::infer_types_dot_general(lhs, rhs, dot_dimension_numbers = ddn)@items
}

# Reduction operations - these reduce dimensions and return lower-rank tensors

.graph_reduce_op <- function(operand, dims, drop) {
  old_shape <- operand@type@shape@dims
  if (drop) {
    new_shape <- old_shape[-dims]
  } else {
    new_shape <- old_shape
    new_shape[dims] <- 1L
  }
  list(stablehlo::ValueType(stablehlo::TensorType(
    operand@type@dtype,
    stablehlo::Shape(new_shape)
  )))
}

p_reduce_sum[["graph"]] <- function(operand, dims, drop) {
  .graph_reduce_op(operand, dims, drop)
}

p_reduce_prod[["graph"]] <- function(operand, dims, drop) {
  .graph_reduce_op(operand, dims, drop)
}

p_reduce_max[["graph"]] <- function(operand, dims, drop) {
  .graph_reduce_op(operand, dims, drop)
}

p_reduce_min[["graph"]] <- function(operand, dims, drop) {
  .graph_reduce_op(operand, dims, drop)
}

p_reduce_any[["graph"]] <- function(operand, dims, drop) {
  # any returns boolean
  old_shape <- operand@type@shape@dims
  if (drop) {
    new_shape <- old_shape[-dims]
  } else {
    new_shape <- old_shape
    new_shape[dims] <- 1L
  }
  list(stablehlo::ValueType(stablehlo::TensorType(
    stablehlo::BooleanType(),
    stablehlo::Shape(new_shape)
  )))
}

p_reduce_all[["graph"]] <- function(operand, dims, drop) {
  # all returns boolean
  old_shape <- operand@type@shape@dims
  if (drop) {
    new_shape <- old_shape[-dims]
  } else {
    new_shape <- old_shape
    new_shape[dims] <- 1L
  }
  list(stablehlo::ValueType(stablehlo::TensorType(
    stablehlo::BooleanType(),
    stablehlo::Shape(new_shape)
  )))
}

# Control flow operations

p_select[["graph"]] <- function(pred, true_value, false_value) {
  stablehlo::infer_types_select(pred, on_true = true_value, on_false = false_value)@items
}

p_if[["graph"]] <- function(pred, true, false) {
  # if is handled specially - it returns the output of the true branch
  # In graph mode, we need to get the output type from the branches
  # For now, just return the true branch output type
  list(true)
}
