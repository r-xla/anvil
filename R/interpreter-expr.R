#' @include array.R
#' @include interpreter.R
#' @include type-inference.R
#' @include box.R
#' @include ir.R
#' @include list_of.R
#' @include expr-builder.R
NULL

ExprInterpreter <- S7::new_class("ExprInterpreter", parent = Interpreter,
  properties = list(
    builder = S7::new_property(S7::class_any, default = Builder$new())
  )
)

ExprBox <- S7::new_class("ExprBox", parent = Box,
  properties = list(
    aval = ShapedArray,
    builder = S7::new_S3_class("Builder")
  )
)

method(aval, ExprBox) <- function(x) {
  x@aval
}

method(process_primitive, ExprBox) <- function(interpreter, prim, args, params) {
  # TODO
}

ExprBoxes <- new_list_of("ExprBoxes", ExprBox)

# To also make things like mul(2, 2) work. In jax this is called omnistaging.
# Because we always stage out the computation, we don't need a dynamic trace,
# because our 'bottom interpreter' is always the stageout interpreter and not an
# eval interpreter.
# We might want to add the eval interpreter later for debugging, but for now I don't care

sa2vt <- function(x) {
  stopifnot(inherits(x, ShapedArray))
  stablehlo::ValueType(stablehlo::TensorType(x@dtype, x@shape))
}

vt2sa <- function(x) {
  stopifnot(inherits(x, stablehlo::ValueType))
  stopifnot(inherits(x@type, stablehlo::TensorType))
  ShapedArray(x@type@dtype, x@type@shape)
}


stageout_generic_biv <- function(lhs, rhs, op) {
  builder <- current_builder()

  # TODO: to really support prim_add(1, 1) we need to lift the abstraction level
  # of the input variables. (omnistaging)

  # 1. Compute the equation. For the shape inference we use the stablehlo inference rules.
  # Assume the inputs are ShapedArrays
  infer_types_generic_biv(sa2vt(lhs), sa2vt(rhs))
  out <- vt2sa(stablehlo::infer_types_generic_biv(lhs, rhs))

  # Now create Equation, which uses Variable Class

  eqn <- Equation(
    primitive = op,
    inputs = list(Variable(lhs), Variable(rhs)),
    params = list(),
    out_binders = list(Variable(out))
  )

  builder$add_equation(eqn)

  return(out)
}

method(op_add, list(class_any, class_any)) <- function(lhs, rhs) {
  stageout_generic_biv(lhs, rhs, op_add)
}
