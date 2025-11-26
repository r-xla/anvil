# Evaluation Interpreter:

EvalInterpreter <- R6Class("")


NULL


make_evaluator_uni <- function(hlo_f, broadcastable) {
  function(operand) {
    type <- as.character(dtype(operand))

    shape <- shape(operand)

    x <- hlo_input("operand", type, shape, func_id = "main")
    z <- hlo_f(x)
    f <- hlo_return(z)

    program <- pjrt_program(repr(f), format = "mlir")
    # TODO: Need to generate client from inputs.
    # Need code in pjrt for this
    exec <- pjrt_compile(program)
    pjrt_execute(exec, operand)
  }
}

make_evaluator_biv <- function(hlo_f, broadcastable) {
  function(lhs, rhs) {
    lhs_type <- as.character(dtype(lhs))
    rhs_type <- as.character(dtype(rhs))

    shape_lhs <- shape(lhs)
    shape_rhs <- shape(rhs)

    x <- hlo_input("lhs", lhs_type, shape_lhs, func_id = "main")
    y <- hlo_input("rhs", rhs_type, shape_rhs)

    if (broadcastable) {
      shape_out <- broadcast_shapes(shape_lhs, shape_rhs)
      if (!identical(shape_lhs, shape_rhs)) {
        bdims_lhs <- seq(
          length(shape_out) - length(shape_lhs),
          length(shape_out) - 1L
        )
        bdims_rhs <- seq(
          length(shape_out) - length(shape_rhs),
          length(shape_out) - 1L
        )
        x <- stablehlo::hlo_broadcast_in_dim(x, bdims_lhs, shape_out)
        y <- stablehlo::hlo_broadcast_in_dim(y, bdims_rhs, shape_out)
      }
    }

    z <- hlo_f(x, y)
    f <- hlo_return(z)

    program <- pjrt_program(repr(f), format = "mlir")
    # TODO: Need to generate client from inputs.
    # Need code in pjrt for this
    exec <- pjrt_compile(program)
    pjrt_execute(exec, lhs, rhs)
  }
}

eval_add <- make_evaluator_biv(hlo_add, TRUE)
eval_abs <- make_evaluator_uni(hlo_abs, TRUE)
eval_abs <- make_evaluator_biv(hlo_abs, TRUE)
eval_matmul <- function(lhs, rhs) {
  lhs_type <- as.character(dtype(lhs))
  rhs_type <- as.character(dtype(rhs))

  shape_lhs <- shape(lhs)
  shape_rhs <- shape(rhs)

  x <- hlo_input("lhs", lhs_type, shape_lhs, func_id = "main")
  y <- hlo_input("rhs", rhs_type, shape_rhs)

  z <- hlo_dot_general(x, y, list(length(shape_lhs) - 1L, 0L))

  f <- hlo_return(z)

  program <- pjrt_program(repr(f), format = "mlir")
  exec <- pjrt_compile(program)
  pjrt_execute(exec, lhs, rhs)
}

EVAL_RULES <- as.environment(list(
  add = eval_add,
  abs = eval_abs,
  matmul = eval_matmul
))
