#' @include tensor.R
#' @include interpreter.R
#' @include type-inference.R
#' @include box.R
#' @include ir.R
#' @include list_of.R
#' @include ir-builder.R
NULL

# TODO: Maybe this thing should contain the builder?
IRInterpreter <- S7::new_class(
  "IRInterpreter",
  parent = Interpreter,
  properties = list(
    builder = S7::new_property(
      S7::new_S3_class("Builder"),
      getter = function(self) {
        self@main@global_data
      },
      setter = function(self, value) {
        if (!identical(self@builder, value)) {
          stop("Can only modify builder in-place")
        }
        self@builder <- value
        invisible(self)
      }
    )
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
    aval = ShapedTensor,
    id = class_environment
  ),
  constructor = function(interpreter, aval) {
    S7::new_object(Box(interpreter), aval = aval, id = new.env(size = 0L))
  }
)

method(ir_id, IRBox) <- function(x) {
  rlang::obj_address(x@id)
}


# Boxing is applied to all variables within a transformed function
# that are captured via lexical scoping and are therefore not an input,
# because all inputs are wrapped in the highest-level box box already
method(box, list(IRInterpreter, class_any)) <- function(interpreter, x) {
  builder <- interpreter@builder
  # nofmt
  box <- interpreter@builder$const_boxes[[ir_id(x)]]
  if (is.null(box)) {
    box <- builder$new_box(interpreter, raise_to_shaped(aval(x)))
    builder$add_constant(box, x)
  }
  return(box)
}

new_arg <- function(interpreter, aval) {
  aval <- raise_to_shaped(aval)
  box <- interpreter@builder$new_box(interpreter, aval)
  interpreter@builder$boxes_to_variables[[ir_id(box)]] <- IRVariable(aval)
  return(box)
}

method(aval, IRBox) <- function(x) {
  x@aval
}


# This is like mlr3torch::LearnerTorch$.train()
method(process_primitive, IRInterpreter) <- function(
  interpreter,
  prim,
  boxes,
  params
) {
  avals_in <- lapply(boxes, aval)
  avals_out <- rlang::exec(prim@ir_rule, !!!c(avals_in, params))
  out_boxes <- lapply(
    avals_out,
    interpreter@builder$new_box,
    interpreter = interpreter
  )
  # translate from tracing world to IR world
  inputs <- lapply(boxes, interpreter@builder$get_variable)
  outvars <- lapply(out_boxes, interpreter@builder$add_variable)
  interpreter@builder$add_equation(IREquation(
    prim,
    IRAtoms(inputs),
    IRParams(params),
    IRVariables(outvars)
  ))
  out_boxes
}

IRBoxes <- new_list_of("IRBoxes", IRBox)

# To also make things like mul(2, 2) work. In jax this is called omnistaging.
# Because we always stage out the computation, we don't need a dynamic trace,
# because our 'bottom interpreter' is always the stageout interpreter and not an
# eval interpreter.
# We might want to add the eval interpreter later for debugging, but for now I don't care

r2ir <- function(f, ...) {
  builder <- Builder$new()
  main <- local_main(IRInterpreter, builder)
  interpreter <- IRInterpreter(main)
  boxes_in <- lapply(list(...), new_arg, interpreter = interpreter)
  outs <- rlang::exec(f, !!!boxes_in)
  # TODO: Why do we need to raise the outputs?
  boxes_out <- lapply(outs, full_raise, interpreter = interpreter)
  # TODO: Handle constants
  res <- builder$build(boxes_in, boxes_out)
  list(
    ir = res[[1]],
    consts = res[[2]]
  )
}
