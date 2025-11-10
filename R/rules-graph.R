# The graph rules are very simple, we do no type inference, but just build the graph and
# make closed over constants explicit, which is e.g. important for the pullback rule.

graph_transform_binary <- function(lhs, rhs) {
  lhs_vt <- st2vt(lhs)
  rhs_vt <- st2vt(rhs)
  lapply(stablehlo::infer_types_generic_biv(lhs_vt, rhs_vt)@items, vt2st)
}

p_add[["graph"]] <- function(lhs, rhs) {
  graph_transform_binary(lhs, rhs)
}

p_mul[["graph"]] <- function(lhs, rhs) {
  graph_transform_binary(lhs, rhs)
}

p_sub[["graph"]] <- function(lhs, rhs) {
  graph_transform_binary(lhs, rhs)
}

p_neg[["graph"]] <- function(lhs, rhs) {
  graph_transform_binary(lhs, rhs)
}

p_div[["graph"]] <- function(lhs, rhs) {
  graph_transform_binary(lhs, rhs)
}

p_pow[["graph"]] <- function(lhs, rhs) {
  graph_transform_binary(lhs, rhs)
}
