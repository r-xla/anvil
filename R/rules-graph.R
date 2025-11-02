# The graph rules are very simple, we do no type inference, but just build the graph and
# make closed over constants explicit, which is e.g. important for the pullback rule.

p_add[["graph"]] <- function(lhs, rhs) {
  lhs_vt <- st2vt(lhs)
  rhs_vt <- st2vt(rhs)
  lapply(stablehlo::infer_types_generic_biv(lhs_vt, rhs_vt)@items, vt2st)
}
