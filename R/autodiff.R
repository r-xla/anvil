backward_pass <- function(ir) {}

grad <- function(f, ...) {
  ir <- stage_out(f, ...) # R -> IR
  ir_backward <- backward(ir) # IR -> IR
  compile(ir_backward) # IR -> (stableHLO -> compiled)
}
