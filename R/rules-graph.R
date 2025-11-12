# The graph rules are very simple, we do no type inference, but just build the graph and
# make closed over constants explicit, which is e.g. important for the pullback rule.

graph_rule_binary <- function(lhs, rhs) {
  stablehlo::infer_types_generic_biv(lhs, rhs)@items
}

#' @importFrom stablehlo infer_types_generic_biv
p_add[["graph"]] <- function(lhs, rhs) {
  graph_rule_binary(lhs, rhs)
}

p_mul[["graph"]] <- function(lhs, rhs) {
  list(infer_types_generic_biv(lhs, rhs))
}

p_sub[["graph"]] <- function(lhs, rhs) {
  list(infer_types_generic_biv(lhs, rhs))
}

p_neg[["graph"]] <- function(lhs, rhs) {
  list(infer_types_generic_biv(lhs, rhs))
}

p_div[["graph"]] <- function(lhs, rhs) {
  list(infer_types_generic_biv(lhs, rhs))
}

p_pow[["graph"]] <- function(lhs, rhs) {
  list(infer_types_generic_biv(lhs, rhs))
}
