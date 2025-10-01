stageout_generic_biv <- function(lhs, rhs, op) {
  outs <- stablehlo::infer_types_generic_biv(st2vt(lhs), st2vt(rhs))
  list(vt2st(outs@items[[1L]]))
}

register_ir_rule(p_add, function(lhs, rhs) {
  stageout_generic_biv(lhs, rhs, nvl_add)
})

register_ir_rule(
  p_broadcast_in_dim,
  function(operand, broadcast_dimensions, shapes_out) {
    # only args are boxed
    outs <- stablehlo:::infer_types_broadcast_in_dim(
      st2vt(operand),
      broadcast_dimensions,
      shapes_out
    )
  }
)
