#' @include tensor.R
#' @include interpreter.R
#' @include type-inference.R
#' @include box.R
#' @include ir.R
#' @include list_of.R
#' @include ir-builder.R
NULL

# TODO: Maybe this thing should contain the builder?
IRInterpreter <- S7::new_class("IRInterpreter",
  parent = Interpreter,
  properties = list(
    builder = S7::new_property(S7::new_S3_class("Builder"), getter = function(self) {
      self@main@global_data
    })
  ),
  constructor = function(main) {
    S7::new_object(Interpreter(main))
  }
)

# This is similar to DynamicJaxprTracer in Jax, but we also store the program
# Builder in the Box, whereas Jax has global data in the MainInterpreter
# This is more in line with the design of both stablehlo and mlr3torch
IRBox <- S7::new_class(
  "IRBox",
  parent = Box,
  properties = list(
    aval = ShapedTensor
  )
)

# Boxing is applied to all variables within a transformed function
# that are captured via lexical scoping and not an input,
# because all inputs are wrapped in the highest-level box box already
method(box, list(IRInterpreter, class_any)) <- function(interpreter, x) {
  builder <- interpreter@builder
  # nofmt
  box <- builder$get(id(x))
  if (is.null(box)) {
    box <- builder$new_box(raise_to_shaped(aval(x)))
    builder$add_constant(box, x)
  }
  return(box)
}

new_arg <- function(interpreter, aval) {
  aval <- raise_to_shaped(aval)
  box <- interpreter@builder$new_box(interpreter, aval)
  interpreter@builder$box_to_var[id(box)] <- IRVariable(aval)
  return(box)
}

method(aval, IRBox) <- function(x) {
  x@var@variable
}


# This is like mlr3torch::LearnerTorch$.train()
method(process_primitive, IRBox) <- function(
  interpreter,
  prim,
  boxes,
  params
) {
  avals_in <- lapply(boxes, aval)
  avals_out <- rlang::exec(prim, !!!c(avals_in, params))

}

IRBoxes <- new_list_of("IRBoxes", IRBox)

# To also make things like mul(2, 2) work. In jax this is called omnistaging.
# Because we always stage out the computation, we don't need a dynamic trace,
# because our 'bottom interpreter' is always the stageout interpreter and not an
# eval interpreter.
# We might want to add the eval interpreter later for debugging, but for now I don't care

sa2vt <- function(x) {
  stopifnot(inherits(x, ShapedTensor))
  stablehlo::ValueType(stablehlo::TensorType(x@dtype, x@shape))
}

vt2sa <- function(x) {
  stopifnot(inherits(x, stablehlo::ValueType))
  stopifnot(inherits(x@type, stablehlo::TensorType))
  ShapedTensor(x@type@dtype, x@type@shape)
}


stageout_generic_biv <- function(lhs, rhs, op) {
  builder <- current_builder()

  # TODO: to really support prim_add(1, 1) we need to lift the abstraction level
  # of the input variables. (omnistaging)

  # 1. Compute the equation. For the shape inference we use the stablehlo inference rules.
  # Assume the inputs are ShapedTensors
  infer_types_generic_biv(sa2vt(lhs), sa2vt(rhs))
  out <- vt2sa(stablehlo::infer_types_generic_biv(lhs, rhs))

  # Now create IREquation, which uses IRVariable Class

  eqn <- IREquation(
    primitive = op,
    inputs = list(IRVariable(lhs), IRVariable(rhs)),
    params = list(),
    out_binders = list(IRVariable(out))
  )

  builder$add_equation(eqn)

  return(out)
}

register_ir_rule(prim_add, function(lhs, rhs) {
  stageout_generic_biv(lhs, rhs, prim_add)
})

lower <- function(f, ...) {
  builder <- Builder$new()
  main <- local_main(IRInterpreter, builder)
  interpreter <- IRInterpreter(main)
  boxes_in <- lapply(list(...), new_arg, interpreter = interpreter)
  outs <- rlang::exec(f, !!!boxes_in)
  # TODO: Why do we need to raise the outputs?
  boxes_out <- lapply(outs, full_raise, interpreter = interpreter)
  # TODO: Handle constants
  ir <- builder$build(boxes_in, boxes_out)
  return(ir)
}
